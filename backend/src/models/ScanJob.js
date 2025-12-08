const mongoose = require('mongoose');

const frameResultSchema = new mongoose.Schema({
  frame_number: Number,
  image: String, // base64
  deepfake_probability: Number,
  confidence: Number
}, { _id: false });

const scanJobSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  jobId: {
    type: String,
    required: true,
    unique: true
  },
  filename: {
    type: String,
    required: true
  },
  fileType: {
    type: String,
    enum: ['video', 'image'],
    required: true
  },
  status: {
    type: String,
    enum: ['pending', 'processing', 'completed', 'failed'],
    default: 'pending'
  },
  result: {
    verdict: {
      type: String,
      enum: ['AUTHENTIC', 'SUSPICIOUS', 'LIKELY_DEEPFAKE', null],
      default: null
    },
    verdict_confidence: String,
    deepfake_score: Number,
    frames_analyzed: Number,
    frames: [frameResultSchema],
    error: String
  },
  processingTime: Number,
  createdAt: {
    type: Date,
    default: Date.now
  },
  completedAt: Date
});

scanJobSchema.index({ userId: 1, createdAt: -1 });
scanJobSchema.index({ jobId: 1 });
scanJobSchema.index({ status: 1 });

module.exports = mongoose.model('ScanJob', scanJobSchema);
