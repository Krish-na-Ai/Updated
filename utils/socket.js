// utils/socket.js
const socketIo = require('socket.io');
const jwt = require('jsonwebtoken');
const User = require('../models/user.model');

// Initialize socket.io
const initializeSocket = (server) => {
  const io = socketIo(server, {
    cors: {
      origin: process.env.FRONTEND_URL || "*",
      methods: ["GET", "POST"],
      credentials: true
    }
  });

  // Authentication middleware for socket
  io.use(async (socket, next) => {
    try {
      const token = socket.handshake.auth.token;
      if (!token) {
        return next(new Error('Authentication error: Token missing'));
      }

      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      const user = await User.findById(decoded.id).select("-password");
      
      if (!user) {
        return next(new Error('Authentication error: User not found'));
      }

      socket.user = user;
      next();
    } catch (error) {
      return next(new Error('Authentication error: ' + error.message));
    }
  });

  // Handle connection
  io.on('connection', (socket) => {
    console.log(`User connected: ${socket.user._id}`);
    
    // Join a room specific to the user
    socket.join(socket.user._id.toString());
    
    // Handle disconnect
    socket.on('disconnect', () => {
      console.log(`User disconnected: ${socket.user._id}`);
    });
  });

  return io;
};

module.exports = initializeSocket;