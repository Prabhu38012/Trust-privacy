const mongoose = require('mongoose');
const crypto = require('crypto');

const certificateSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  mediaHash: {
    type: String,
    required: true,
    unique: true
  },
  mediaType: {
    type: String,
    enum: ['image', 'video', 'audio', 'document'],
    required: true
  },
  originalFilename: {
    type: String,
    required: true
  },
  certificateId: {
    type: String,
    unique: true,
    default: () => `TL-${Date.now().toString(36).toUpperCase()}-${crypto.randomBytes(4).toString('hex').toUpperCase()}`
  },
  blockchainTxHash: {
    type: String,
    default: null
  },
  verificationStatus: {
    type: String,
    enum: ['pending', 'verified', 'failed', 'revoked'],
    default: 'pending'
  },
  deepfakeScore: {
    type: Number,
    min: 0,
    max: 100,
    default: null
  },
  metadata: {
    fileSize: Number,
    dimensions: {
      width: Number,
      height: Number
    },
    duration: Number,
    format: String
  },
  issuedAt: {
    type: Date,
    default: Date.now
  },
  expiresAt: {
    type: Date,
    default: () => new Date(Date.now() + 365 * 24 * 60 * 60 * 1000) // 1 year
  }
}, {
  timestamps: true
});

// Indexes
certificateSchema.index({ userId: 1, createdAt: -1 });
certificateSchema.index({ certificateId: 1 });
certificateSchema.index({ mediaHash: 1 });

module.exports = mongoose.model('Certificate', certificateSchema);
