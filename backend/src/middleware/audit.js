const AuditLog = require('../models/AuditLog');

const logAction = async (userId, action, details = {}, req = null) => {
  try {
    await AuditLog.create({
      userId,
      action,
      details,
      ip: req?.ip || req?.connection?.remoteAddress || 'unknown',
      userAgent: req?.headers?.['user-agent'] || 'unknown'
    });
  } catch (error) {
    console.error('Failed to log audit action:', error);
  }
};

// Middleware for automatic audit logging
const auditMiddleware = (action) => {
  return (req, res, next) => {
    // Store original json method
    const originalJson = res.json.bind(res);
    
    res.json = (data) => {
      // Log after successful response
      if (res.statusCode >= 200 && res.statusCode < 300 && req.userId) {
        logAction(req.userId, action, { path: req.path }, req);
      }
      return originalJson(data);
    };
    
    next();
  };
};

module.exports = { logAction, auditMiddleware };
