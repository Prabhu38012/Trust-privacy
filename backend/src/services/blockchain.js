const { ethers } = require('ethers');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Load contract config
let contractConfig = { address: '', chainId: '31337', network: 'localhost', rpcUrl: 'http://127.0.0.1:8545' };
const configPath = path.join(__dirname, '../config/contract.json');
if (fs.existsSync(configPath)) {
  contractConfig = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  console.log('📄 Contract config loaded:', contractConfig.address);
}

// Load ABI
let contractABI = [];
const abiPath = path.join(__dirname, '../../blockchain/deployments/AuthCert.abi.json');
const localAbiPath = path.join(__dirname, '../config/AuthCert.abi.json');
if (fs.existsSync(abiPath)) {
  contractABI = JSON.parse(fs.readFileSync(abiPath, 'utf8'));
} else if (fs.existsSync(localAbiPath)) {
  contractABI = JSON.parse(fs.readFileSync(localAbiPath, 'utf8'));
}

// Minimal ABI if file not found
if (contractABI.length === 0) {
  contractABI = [
    "function issueCertificate(bytes32 fileHash, string calldata verdict, uint8 score) external returns (bytes32)",
    "function verifyCertificate(bytes32 certId) external view returns (bytes32, string, uint8, uint256, address)",
    "function certificateExists(bytes32 certId) external view returns (bool)",
    "function totalCertificates() external view returns (uint256)",
    "event CertificateIssued(bytes32 indexed certId, bytes32 indexed fileHash, string verdict, uint8 score, uint256 timestamp, address indexed issuer)"
  ];
}

// Network configs - add localhost
const NETWORKS = {
  '31337': {
    name: 'Localhost',
    rpcUrl: 'http://127.0.0.1:8545',
    explorer: 'http://localhost:8545',
    currency: 'ETH'
  },
  '80001': {
    name: 'Mumbai',
    rpcUrl: 'https://rpc-mumbai.maticvigil.com',
    explorer: 'https://mumbai.polygonscan.com',
    currency: 'MATIC'
  },
  '11155111': {
    name: 'Sepolia',
    rpcUrl: 'https://rpc.sepolia.org',
    explorer: 'https://sepolia.etherscan.io',
    currency: 'ETH'
  }
};

class BlockchainService {
  constructor() {
    this.provider = null;
    this.wallet = null;
    this.contract = null;
    this.initialized = false;
    this.networkConfig = NETWORKS[contractConfig.chainId] || NETWORKS['31337'];
  }

  /**
   * Initialize blockchain connection
   */
  async initialize() {
    if (this.initialized) return true;

    try {
      // Check if we have contract address and private key
      if (!contractConfig.address) {
        console.log('⚠️ No contract address configured');
        return false;
      }

      if (!process.env.BLOCKCHAIN_PRIVATE_KEY) {
        console.log('⚠️ No blockchain private key configured');
        return false;
      }

      // Setup provider
      const rpcUrl = process.env.RPC_URL || this.networkConfig.rpcUrl;
      this.provider = new ethers.JsonRpcProvider(rpcUrl);

      // Setup wallet
      this.wallet = new ethers.Wallet(process.env.BLOCKCHAIN_PRIVATE_KEY, this.provider);
      console.log(`🔗 Blockchain wallet: ${this.wallet.address}`);

      // Setup contract
      this.contract = new ethers.Contract(contractConfig.address, contractABI, this.wallet);
      console.log(`📜 Contract address: ${contractConfig.address}`);

      // Verify connection
      const network = await this.provider.getNetwork();
      console.log(`🌐 Connected to: ${this.networkConfig.name} (Chain ID: ${network.chainId})`);

      this.initialized = true;
      return true;
    } catch (error) {
      console.error('❌ Blockchain initialization failed:', error.message);
      return false;
    }
  }

  /**
   * Compute SHA256 hash of file buffer
   */
  computeFileHash(fileBuffer) {
    const hash = crypto.createHash('sha256').update(fileBuffer).digest('hex');
    return '0x' + hash;
  }

  /**
   * Issue a certificate on-chain
   */
  async issueCertificate(fileBuffer, verdict, score) {
    if (!this.initialized) {
      const init = await this.initialize();
      if (!init) {
        throw new Error('Blockchain not initialized');
      }
    }

    try {
      // Compute file hash
      const fileHash = this.computeFileHash(fileBuffer);
      console.log(`📁 File hash: ${fileHash}`);

      // Ensure score is uint8 (0-100)
      const scoreInt = Math.min(100, Math.max(0, Math.round(score)));

      // Issue certificate
      console.log('📝 Issuing certificate on-chain...');
      const tx = await this.contract.issueCertificate(fileHash, verdict, scoreInt);
      console.log(`📤 Transaction sent: ${tx.hash}`);

      // Wait for confirmation
      const receipt = await tx.wait();
      console.log(`✅ Transaction confirmed in block ${receipt.blockNumber}`);

      // Parse certificate ID from event
      let certId = null;
      for (const log of receipt.logs) {
        try {
          const parsed = this.contract.interface.parseLog(log);
          if (parsed && parsed.name === 'CertificateIssued') {
            certId = parsed.args.certId;
            break;
          }
        } catch (e) {
          // Not our event, skip
        }
      }

      return {
        success: true,
        transactionHash: tx.hash,
        blockNumber: receipt.blockNumber,
        certificateId: certId,
        fileHash,
        explorerUrl: `${this.networkConfig.explorer}/tx/${tx.hash}`,
        network: this.networkConfig.name
      };
    } catch (error) {
      console.error('❌ Certificate issuance failed:', error);
      throw error;
    }
  }

  /**
   * Verify a certificate by ID - fetch from blockchain
   */
  async verifyCertificate(certId) {
    if (!this.initialized) {
      const init = await this.initialize();
      if (!init) return { exists: false };
    }

    try {
      // Handle both string and bytes32 formats
      let certIdBytes = certId;
      if (typeof certId === 'string' && !certId.startsWith('0x')) {
        // If it's a UUID, hash it to get bytes32
        const crypto = require('crypto');
        certIdBytes = '0x' + crypto.createHash('sha256').update(certId).digest('hex');
      }

      const exists = await this.contract.certificateExists(certIdBytes);
      if (!exists) {
        return { exists: false };
      }

      const result = await this.contract.verifyCertificate(certIdBytes);
      return {
        exists: true,
        fileHash: result[0],
        verdict: result[1],
        score: Number(result[2]),
        timestamp: Number(result[3]),
        issuer: result[4]
      };
    } catch (error) {
      console.error('Certificate verification error:', error.message);
      return { exists: false, error: error.message };
    }
  }

  /**
   * Check if a certificate exists
   */
  async certificateExists(certId) {
    if (!this.initialized) {
      await this.initialize();
    }
    return await this.contract.certificateExists(certId);
  }

  /**
   * Get total certificates count
   */
  async getTotalCertificates() {
    if (!this.initialized) {
      await this.initialize();
    }
    const total = await this.contract.totalCertificates();
    return Number(total);
  }

  /**
   * Get wallet balance
   */
  async getBalance() {
    if (!this.initialized) {
      await this.initialize();
    }
    const balance = await this.provider.getBalance(this.wallet.address);
    return ethers.formatEther(balance);
  }

  /**
   * Get explorer URL for a transaction
   */
  getExplorerUrl(txHash) {
    return `${this.networkConfig.explorer}/tx/${txHash}`;
  }

  /**
   * Get explorer URL for a certificate
   */
  getCertificateExplorerUrl(txHash) {
    return `${this.networkConfig.explorer}/tx/${txHash}`;
  }
}

// Export singleton instance
module.exports = new BlockchainService();
