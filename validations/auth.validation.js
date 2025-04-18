const Joi = require("joi");

// Make sure Joi is properly installed
if (!Joi) {
  console.error('Joi is not properly installed or imported');
}

const signupSchema = Joi.object({
  email: Joi.string().email().lowercase().required()
    .messages({
      "string.email": "Please enter a valid email address",
      "any.required": "Email is required"
    }),

  password: Joi.string()
    .pattern(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%?&])[A-Za-z\d@$!%?&]{8,}$/)
    .required()
    .messages({
      "string.pattern.base": "Password must contain at least 8 characters, including uppercase, lowercase, number and special character"
    }),

  confirmPassword: Joi.string()
    .valid(Joi.ref('password'))
    .required()
    .messages({ "any.only": "Passwords do not match" }),

  name: Joi.string().min(2).max(50).required()
    .messages({ "string.min": "Name must be at least 2 characters" }),

  phone: Joi.string()
    .pattern(/^(\+\d{1,3}[- ]?)?\d{10,14}$/)
    .required()
    .messages({
      "string.pattern.base": "Please enter a valid phone number (10-14 digits with optional country code)"
    })
});


const loginSchema = Joi.object({
  email: Joi.string().email().lowercase().required(),
  password: Joi.string().required()
});

// Check that schemas are defined before exporting
console.log('Signup schema defined:', !!signupSchema);
console.log('Login schema defined:', !!loginSchema);

// Export the schemas
module.exports = {
  signupSchema,
  loginSchema
};