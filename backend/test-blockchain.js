require('dotenv').config();
const blockchainService = require('./src/services/blockchain');

async function test() {
  console.log('🧪 Testing blockchain connection...\n');
  
  try {
    // Initialize
    const initialized = await blockchainService.initialize();
    console.log('Initialized:', initialized);
    
    if (!initialized) {
      console.log('❌ Blockchain not initialized. Check your .env and contract.json');
      return;
    }
    
    // Get balance
    const balance = await blockchainService.getBalance();
    console.log('Wallet balance:', balance, 'ETH');
    
    // Get total certificates
    const total = await blockchainService.getTotalCertificates();
    console.log('Total certificates:', total);
    
    // Issue a test certificate
    console.log('\n📝 Issuing test certificate...');
    const testHash = Buffer.from('test-file-hash-' + Date.now());
    const result = await blockchainService.issueCertificate(testHash, 'AUTHENTIC', 25);
    
    console.log('\n✅ Certificate issued!');
    console.log('Transaction:', result.transactionHash);
    console.log('Certificate ID:', result.certificateId);
    
    // Verify
    const newTotal = await blockchainService.getTotalCertificates();
    console.log('\nTotal certificates now:', newTotal);
    
    console.log('\n🎉 Blockchain is working!');
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

test();
