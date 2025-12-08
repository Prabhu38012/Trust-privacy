// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title AuthCert - TrustLock Authenticity Certificate
 * @notice Stores deepfake analysis results on-chain for verification
 * @dev Each certificate contains file hash, verdict, and timestamp
 */
contract AuthCert {
    
    struct Certificate {
        bytes32 fileHash;       // SHA256 hash of the analyzed file
        string verdict;         // AUTHENTIC, SUSPICIOUS, LIKELY_DEEPFAKE, etc.
        uint8 score;            // Deepfake score (0-100)
        uint256 timestamp;      // When the analysis was performed
        address issuer;         // Who issued the certificate
        bool exists;            // Whether this certificate exists
    }
    
    // Mapping from certificate ID to Certificate
    mapping(bytes32 => Certificate) public certificates;
    
    // Array of all certificate IDs for enumeration
    bytes32[] public certificateIds;
    
    // Owner of the contract
    address public owner;
    
    // Authorized issuers
    mapping(address => bool) public authorizedIssuers;
    
    // Events
    event CertificateIssued(
        bytes32 indexed certId,
        bytes32 indexed fileHash,
        string verdict,
        uint8 score,
        uint256 timestamp,
        address indexed issuer
    );
    
    event IssuerAuthorized(address indexed issuer);
    event IssuerRevoked(address indexed issuer);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }
    
    modifier onlyAuthorized() {
        require(authorizedIssuers[msg.sender] || msg.sender == owner, "Not authorized");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        authorizedIssuers[msg.sender] = true;
    }
    
    /**
     * @notice Issue a new certificate for a file analysis
     * @param fileHash SHA256 hash of the analyzed file
     * @param verdict The analysis verdict (AUTHENTIC, SUSPICIOUS, etc.)
     * @param score The deepfake probability score (0-100)
     * @return certId The unique certificate ID
     */
    function issueCertificate(
        bytes32 fileHash,
        string calldata verdict,
        uint8 score
    ) external onlyAuthorized returns (bytes32 certId) {
        require(score <= 100, "Score must be 0-100");
        require(bytes(verdict).length > 0, "Verdict cannot be empty");
        
        // Generate unique certificate ID
        certId = keccak256(abi.encodePacked(
            fileHash,
            block.timestamp,
            msg.sender,
            certificateIds.length
        ));
        
        // Ensure certificate doesn't already exist
        require(!certificates[certId].exists, "Certificate already exists");
        
        // Store certificate
        certificates[certId] = Certificate({
            fileHash: fileHash,
            verdict: verdict,
            score: score,
            timestamp: block.timestamp,
            issuer: msg.sender,
            exists: true
        });
        
        certificateIds.push(certId);
        
        emit CertificateIssued(certId, fileHash, verdict, score, block.timestamp, msg.sender);
        
        return certId;
    }
    
    /**
     * @notice Verify a certificate by its ID
     * @param certId The certificate ID to verify
     * @return fileHash The file hash
     * @return verdict The verdict
     * @return score The score
     * @return timestamp When it was issued
     * @return issuer Who issued it
     */
    function verifyCertificate(bytes32 certId) external view returns (
        bytes32 fileHash,
        string memory verdict,
        uint8 score,
        uint256 timestamp,
        address issuer
    ) {
        require(certificates[certId].exists, "Certificate does not exist");
        Certificate memory cert = certificates[certId];
        return (cert.fileHash, cert.verdict, cert.score, cert.timestamp, cert.issuer);
    }
    
    /**
     * @notice Check if a certificate exists
     * @param certId The certificate ID to check
     * @return exists Whether the certificate exists
     */
    function certificateExists(bytes32 certId) external view returns (bool) {
        return certificates[certId].exists;
    }
    
    /**
     * @notice Find certificates by file hash
     * @param fileHash The file hash to search for
     * @return matchingCerts Array of certificate IDs for this file
     */
    function findByFileHash(bytes32 fileHash) external view returns (bytes32[] memory) {
        uint256 count = 0;
        
        // First pass: count matches
        for (uint256 i = 0; i < certificateIds.length; i++) {
            if (certificates[certificateIds[i]].fileHash == fileHash) {
                count++;
            }
        }
        
        // Second pass: collect matches
        bytes32[] memory matches = new bytes32[](count);
        uint256 index = 0;
        for (uint256 i = 0; i < certificateIds.length; i++) {
            if (certificates[certificateIds[i]].fileHash == fileHash) {
                matches[index] = certificateIds[i];
                index++;
            }
        }
        
        return matches;
    }
    
    /**
     * @notice Get total number of certificates issued
     * @return count Total certificates
     */
    function totalCertificates() external view returns (uint256) {
        return certificateIds.length;
    }
    
    /**
     * @notice Authorize a new issuer
     * @param issuer Address to authorize
     */
    function authorizeIssuer(address issuer) external onlyOwner {
        authorizedIssuers[issuer] = true;
        emit IssuerAuthorized(issuer);
    }
    
    /**
     * @notice Revoke an issuer's authorization
     * @param issuer Address to revoke
     */
    function revokeIssuer(address issuer) external onlyOwner {
        authorizedIssuers[issuer] = false;
        emit IssuerRevoked(issuer);
    }
    
    /**
     * @notice Transfer contract ownership
     * @param newOwner New owner address
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
