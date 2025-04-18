const express = require("express");
const { signup, login, logout, getProfile } = require("../controllers/auth.controller");
const validate = require("../middlewares/validate.middleware");
const validationSchemas = require("../validations/auth.validation");
const { protect } = require("../middlewares/auth.middleware");

// Log the imported schemas to verify they exist
console.log('Auth validation schemas:', validationSchemas);

const router = express.Router();
router.post("/signup", validate(validationSchemas.signupSchema), signup);
router.post("/login", validate(validationSchemas.loginSchema), login);
router.post("/logout", logout);
router.get("/profile", protect, getProfile);

module.exports = router;