const User = require("../models/user.model");
const { signToken } = require("../utils/jwt");

exports.signup = async (req, res) => {
  const { email, password, name, phone } = req.body;
  // Note: confirmPassword is validated by Joi but not needed in DB
  try {
    const existing = await User.findOne({ email });
    if (existing) return res.status(400).json({ message: "Email already in use" });

    const phoneExists = await User.findOne({ phone });
    if (phoneExists) return res.status(400).json({ message: "Phone number already in use" });

    const user = await User.create({ email, password, name, phone });
    const token = signToken(user);
    
    // Don't return the password hash
    const userResponse = {
      id: user._id,
      name: user.name,
      email: user.email,
      phone: user.phone
    };
    
    res.status(201).json({ token, user: userResponse });
  } catch (err) {
    res.status(500).json({ message: "Signup error", error: err.message });
  }
};

exports.login = async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = await User.findOne({ email });
    if (!user || !(await user.comparePassword(password))) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    const token = signToken(user);
    
    const userResponse = {
      id: user._id,
      name: user.name,
      email: user.email,
      phone: user.phone
    };
    
    res.json({ token, user: userResponse });
  } catch (err) {
    res.status(500).json({ message: "Login failed", error: err.message });
  }
};

exports.logout = (_, res) => {
  res.json({ message: "Logout handled client-side" });
};

exports.getProfile = async (req, res) => {
  try {
    const user = await User.findById(req.user._id).select("-password");
    if (!user) return res.status(404).json({ message: "User not found" });
    res.json(user);
  } catch (err) {
    res.status(500).json({ message: "Error fetching profile", error: err.message });
  }
};