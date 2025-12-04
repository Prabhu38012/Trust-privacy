const mongoose = require('mongoose');

const deepfakeScanSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  mediaHash: {
    type: String,
    required: true
  },
  mediaType: {
    type: String,
    enum: ['image', 'video', 'audio'],
    required: true
  },
  scanResult: {
    isDeepfake: {
      type: Boolean,
      default: false
    },
    confidence: {
      type: Number,
      min: 0,
      max: 100,
      required: true
    },
    detectionMethod: {
      type: String,
      enum: ['facial_analysis', 'artifact_detection', 'audio_analysis', 'metadata_analysis'],
      required: true
    },
    details: {
      type: mongoose.Schema.Types.Mixed,
      default: {}
    }
  },
  processingTime: {
    type: Number, // milliseconds
    required: true
  },
  status: {
    type: String,
    enum: ['processing', 'completed', 'failed'],
    default: 'processing'
  }
}, {
  timestamps: true
});

// Indexes
deepfakeScanSchema.index({ userId: 1, createdAt: -1 });
deepfakeScanSchema.index({ status: 1 });

module.exports = mongoose.model('DeepfakeScan', deepfakeScanSchema);
