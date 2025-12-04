const crypto = require('crypto');

class BlockchainService {
  constructor() {
    this.chain = [];
    this.pendingCertificates = [];
    // Create genesis block
    this.createBlock(1, '0');
  }

  createBlock(nonce, previousHash) {
    const block = {
      index: this.chain.length + 1,
      timestamp: Date.now(),
      certificates: this.pendingCertificates,
      nonce: nonce,
      previousHash: previousHash,
      hash: this.calculateHash(this.chain.length + 1, previousHash, nonce),
    };
    
    this.pendingCertificates = [];
    this.chain.push(block);
    return block;
  }

  calculateHash(index, previousHash, nonce) {
    const data = index + previousHash + nonce + JSON.stringify(this.pendingCertificates);
    return crypto.createHash('sha256').update(data).digest('hex');
  }

  getLastBlock() {
    return this.chain[this.chain.length - 1];
  }

  minePendingCertificates() {
    const lastBlock = this.getLastBlock();
    let nonce = 0;
    let hash = '';
    
    // Simple proof-of-work (find hash starting with '0000')
    while (!hash.startsWith('0000')) {
      nonce++;
      hash = this.calculateHash(this.chain.length + 1, lastBlock.hash, nonce);
    }
    
    return this.createBlock(nonce, lastBlock.hash);
  }

  addCertificate(certificate) {
    this.pendingCertificates.push({
      id: certificate.id,
      scanId: certificate.scanId,
      userId: certificate.userId,
      verdict: certificate.verdict,
      timestamp: Date.now(),
      hash: this.hashCertificate(certificate),
    });

    // Auto-mine after 5 certificates
    if (this.pendingCertificates.length >= 5) {
      return this.minePendingCertificates();
    }
    return null;
  }

  hashCertificate(cert) {
    const data = JSON.stringify({
      id: cert.id,
      scanId: cert.scanId,
      verdict: cert.verdict,
    });
    return crypto.createHash('sha256').update(data).digest('hex');
  }

  verifyCertificate(certificateHash) {
    for (const block of this.chain) {
      for (const cert of block.certificates) {
        if (cert.hash === certificateHash) {
          return {
            valid: true,
            block: block.index,
            timestamp: cert.timestamp,
            certificate: cert,
          };
        }
      }
    }
    return { valid: false };
  }

  isChainValid() {
    for (let i = 1; i < this.chain.length; i++) {
      const currentBlock = this.chain[i];
      const previousBlock = this.chain[i - 1];

      // Verify hash
      const recalculatedHash = this.calculateHash(
        currentBlock.index,
        currentBlock.previousHash,
        currentBlock.nonce
      );

      if (currentBlock.hash !== recalculatedHash) {
        return false;
      }

      // Verify chain link
      if (currentBlock.previousHash !== previousBlock.hash) {
        return false;
      }
    }
    return true;
  }
}

// Singleton instance
const blockchain = new BlockchainService();

module.exports = blockchain;
