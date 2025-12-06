const mongoose = require('mongoose');

const certificateSchema = new mongoose.Schema({
  id: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  scanId: {
    type: String,
    required: true
  },
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  filename: {
    type: String,
    required: true
  },
  fileHash: {
    type: String,
    required: true
  },
  verdict: {
    type: String,
    enum: ['AUTHENTIC', 'LIKELY_AUTHENTIC', 'UNCERTAIN', 'SUSPICIOUS', 'LIKELY_DEEPFAKE'],
    required: true
  },
  score: {
    type: Number,
    required: true,
    min: 0,
    max: 100
  },
  timestamp: {
    type: Date,
    default: Date.now
  },
  blockchain: {
    onChain: {
      type: Boolean,
      default: false
    },
    transactionHash: String,
    blockNumber: Number,
    network: String,
    explorerUrl: String,
    reason: String
  }
}, {
  timestamps: true
});

// Indexes
certificateSchema.index({ scanId: 1 });
certificateSchema.index({ userId: 1, timestamp: -1 });
certificateSchema.index({ fileHash: 1 });

module.exports = mongoose.model('Certificate', certificateSchema);
