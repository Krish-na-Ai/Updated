const jwt = require("jsonwebtoken");
exports.signToken = (user) => jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: "7d" });