const mongoose = require('mongoose');

const auditLogSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  action: {
    type: String,
    required: true,
    enum: [
      'USER_SIGNUP',
      'USER_LOGIN',
      'USER_LOGOUT',
      'PASSWORD_CHANGE',
      'CONSENT_UPDATE',
      'DATA_EXPORT_REQUEST',
      'DATA_DELETE_REQUEST',
      'NOTE_CREATE',
      'NOTE_DELETE',
      'PROFILE_UPDATE',
      'ADMIN_ACCESS',
      'DEEPFAKE_SCAN_START',
      'DEEPFAKE_SCAN_COMPLETE',
      'CERTIFICATE_CREATE',
      'CERTIFICATE_VERIFY'
    ]
  },
  details: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  ip: {
    type: String
  },
  userAgent: {
    type: String
  }
}, {
  timestamps: true
});

// Index for efficient queries
auditLogSchema.index({ userId: 1, createdAt: -1 });
auditLogSchema.index({ action: 1, createdAt: -1 });

module.exports = mongoose.model('AuditLog', auditLogSchema);
