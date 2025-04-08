const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const JWT_SECRET = 'your-secret-key'; 

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend')));

// MongoDB Connection Configuration
const connectWithRetry = async () => {
    // Try local MongoDB first
    try {
        await mongoose.connect('mongodb://127.0.0.1:27017/userAuthDB', {
            useNewUrlParser: true,
            useUnifiedTopology: true,
            serverSelectionTimeoutMS: 5000 // Timeout after 5s
        });
        console.log('Connected to local MongoDB successfully');
        return;
    } catch (err) {
        console.warn('Could not connect to local MongoDB:', err.message);
        console.log('Trying MongoDB Atlas as fallback...');
    }

    // If local fails, try MongoDB Atlas
    try {
        // Free MongoDB Atlas connection string - replace with your own in production
        const atlasUri = "mongodb+srv://demo:demo12345@cluster0.mongodb.net/userAuthDB?retryWrites=true&w=majority";
        await mongoose.connect(atlasUri, {
            useNewUrlParser: true,
            useUnifiedTopology: true
        });
        console.log('Connected to MongoDB Atlas successfully');
    } catch (err) {
        console.error('MongoDB connection error:', err);
        console.error('Could not connect to MongoDB. Please check your connection.');
    }
};

// Connect to MongoDB with retry
connectWithRetry();

// Handle MongoDB connection events
mongoose.connection.on('connected', () => {
    console.log('Mongoose connected to MongoDB');
});

mongoose.connection.on('error', (err) => {
    console.error('Mongoose connection error:', err);
});

mongoose.connection.on('disconnected', () => {
    console.log('Mongoose disconnected from MongoDB');
});

// User Schema
const userSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    createdAt: { type: Date, default: Date.now }
});
const User = mongoose.model('User', userSchema);

// AI Test Response Schema
const aiTestResponseSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    responses: { type: Array, required: true },
    createdAt: { type: Date, default: Date.now }
});

// Create a model for the collection
const AiTestResponse = mongoose.model('AiTestResponse', aiTestResponseSchema, 'testcollection');

app.post('/api/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        // Check if user exists
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(400).json({ message: 'Invalid email or password' });
        }

        // Check password
        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) {
            return res.status(400).json({ message: 'Invalid email or password' });
        }

        // Generate token
        const token = jwt.sign(
            { id: user._id, email: user.email, name: user.name },
            JWT_SECRET,
            { expiresIn: '1h' }
        );

        res.status(200).json({
            token,
            user: { id: user._id, name: user.name, email: user.email }
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// Middleware to authenticate token
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) return res.status(401).json({ message: 'Access denied' });

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) return res.status(403).json({ message: 'Invalid or expired token' });
        req.user = user;
        next();
    });
};

app.post('/api/signup', async (req, res) => {
    try {
        const { name, email, password } = req.body;

        const existingUser = await User.findOne({ email });
        if (existingUser) return res.status(400).json({ message: "User already exists" });

        const hashedPassword = await bcrypt.hash(password, 10);
        const newUser = new User({ name, email, password: hashedPassword });
        await newUser.save();

        res.status(201).json({ message: "User registered successfully!" });
    } catch (error) {
        console.error('Signup error:', error);
        res.status(500).json({ message: "Server error" });
    }
});

app.post('/api/save-ai-test', async (req, res) => {
    try {
        const { name, email, responses } = req.body;

        // Check if the user already exists
        const existingResponse = await AiTestResponse.findOne({ email });
        if (existingResponse) {
            return res.status(400).json({ message: 'Response already exists for this email.' });
        }

        // Save the response to MongoDB
        const newResponse = new AiTestResponse({ name, email, responses });
        await newResponse.save();

        res.status(201).json({ message: 'AI Test response saved successfully!' });
    } catch (error) {
        console.error('Error saving AI Test response:', error);
        res.status(500).json({ message: 'Failed to save AI Test response' });
    }
});

// Route for redirecting to dashboard after login
app.get('/dashboard', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/dashboard.html'));
});

// Serve frontend static files
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Access the application at http://localhost:${PORT}`);
});
