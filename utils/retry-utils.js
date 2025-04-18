// utils/retry-utils.js
/**
 * Retries a function with exponential backoff
 * @param {Function} fn - The function to retry
 * @param {Object} options - Options for retry
 * @param {number} options.maxRetries - Maximum number of retries (default: 3)
 * @param {number} options.initialDelay - Initial delay in ms (default: 300)
 * @param {number} options.maxDelay - Maximum delay in ms (default: 3000)
 * @returns {Promise} - Promise resolving to the function result
 */
exports.withRetry = async (fn, options = {}) => {
    const maxRetries = options.maxRetries || 3;
    const initialDelay = options.initialDelay || 300;
    const maxDelay = options.maxDelay || 3000;
    
    let retries = 0;
    let delay = initialDelay;
    
    while (retries <= maxRetries) {
      try {
        return await fn();
      } catch (error) {
        retries++;
        if (retries > maxRetries) {
          throw error;
        }
        
        console.log(`Retry attempt ${retries}/${maxRetries} after ${delay}ms`);
        
        // Wait for the delay
        await new Promise(resolve => setTimeout(resolve, delay));
        
        // Exponential backoff with jitter
        delay = Math.min(delay * 2, maxDelay) * (0.9 + Math.random() * 0.2);
      }
    }
  };