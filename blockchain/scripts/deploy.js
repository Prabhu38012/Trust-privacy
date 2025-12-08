const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🚀 Deploying AuthCert contract...\n");

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(balance), "ETH/MATIC\n");

  // Deploy contract
  const AuthCert = await hre.ethers.getContractFactory("AuthCert");
  const authCert = await AuthCert.deploy();
  await authCert.waitForDeployment();

  const contractAddress = await authCert.getAddress();
  console.log("✅ AuthCert deployed to:", contractAddress);

  // Get network info
  const network = await hre.ethers.provider.getNetwork();
  console.log("Network:", network.name, "(Chain ID:", network.chainId.toString(), ")");

  // Save deployment info
  const deploymentInfo = {
    contractAddress,
    network: network.name,
    chainId: network.chainId.toString(),
    deployer: deployer.address,
    deployedAt: new Date().toISOString(),
    blockNumber: await hre.ethers.provider.getBlockNumber()
  };

  // Save to file
  const deploymentsDir = path.join(__dirname, "../deployments");
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir, { recursive: true });
  }

  const deploymentPath = path.join(deploymentsDir, `${network.name || 'unknown'}.json`);
  fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));
  console.log("\n📄 Deployment info saved to:", deploymentPath);

  // Also save ABI for backend
  const artifactPath = path.join(__dirname, "../artifacts/contracts/AuthCert.sol/AuthCert.json");
  if (fs.existsSync(artifactPath)) {
    const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
    const abiPath = path.join(deploymentsDir, "AuthCert.abi.json");
    fs.writeFileSync(abiPath, JSON.stringify(artifact.abi, null, 2));
    console.log("📄 ABI saved to:", abiPath);
  }

  // Copy to backend
  const backendConfigDir = path.join(__dirname, "../../backend/src/config");
  if (fs.existsSync(backendConfigDir)) {
    fs.writeFileSync(
      path.join(backendConfigDir, "contract.json"),
      JSON.stringify({
        address: contractAddress,
        chainId: network.chainId.toString(),
        network: network.name
      }, null, 2)
    );
    console.log("📄 Contract config copied to backend");
  }

  // Explorer URL
  const explorers = {
    "80001": "https://mumbai.polygonscan.com",
    "11155111": "https://sepolia.etherscan.io"
  };
  const explorer = explorers[network.chainId.toString()];
  if (explorer) {
    console.log(`\n🔍 View on explorer: ${explorer}/address/${contractAddress}`);
  }

  console.log("\n✅ Deployment complete!");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
