const jwt = require('jsonwebtoken');

const auth = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      console.log('[Auth] No token provided');
      return res.status(401).json({ message: 'Access denied. No token provided.' });
    }

    const token = authHeader.split(' ')[1];
    
    if (!token || token === 'null' || token === 'undefined') {
      console.log('[Auth] Invalid token format');
      return res.status(401).json({ message: 'Invalid token.' });
    }

    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      req.userId = decoded.id;
      req.user = { id: decoded.id };
      next();
    } catch (jwtError) {
      console.log('[Auth] JWT Error:', jwtError.message);
      
      if (jwtError.name === 'TokenExpiredError') {
        return res.status(401).json({ message: 'Token expired. Please login again.', code: 'TOKEN_EXPIRED' });
      }
      
      return res.status(401).json({ message: 'Invalid token.' });
    }
  } catch (error) {
    console.error('[Auth] Error:', error.message);
    return res.status(500).json({ message: 'Authentication error.' });
  }
};

const adminOnly = (req, res, next) => {
  if (req.user?.role !== 'admin') {
    return res.status(403).json({ message: 'Admin access required.' });
  }
  next();
};

module.exports = { auth, adminOnly };
