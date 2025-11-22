# Security Key Management

This document defines the secure handling strategy for TLS certificates, private keys, and AWS IoT credentials for AxioBionics.

## Objectives
- Eliminate hard‑coded / committed credentials.
- Support per-environment provisioning (development, staging, production).
- Use platform-native secure storage when available.
- Enable rotation without code changes.

## Sources of Secrets
- AWS IoT: Device certificate, private key, AmazonRootCA.
- MQTT Auth (non-iOS): May use client certs or credentials.
- Future: API tokens, patient data encryption keys.

## Storage Strategy by Platform
| Platform | Recommended Storage | Access Pattern |
|----------|---------------------|----------------|
| macOS/Linux | File paths via env variables (`AXIO_CERT_PATH`, `AXIO_KEY_PATH`, `AXIO_CA_PATH`) with permissions `600` | Read at startup, load into `QSslConfiguration` |
| iOS | iOS Keychain (imported via configuration tool or MDM) | Query by key label, convert to `SecIdentity` |
| Android | Android Keystore (Hardware-backed if possible) | Use `KeyStore` APIs; wrap with JNI for Qt |
| Embedded (Pi) | Mounted secure partition or TPM-backed store | Direct file read or TPM API |

## Environment Variables
Define the following (example `zsh`):
```bash
export AXIO_CERT_PATH="$HOME/.axio/keys/device.cert.pem"
export AXIO_KEY_PATH="$HOME/.axio/keys/device.private.pem"
export AXIO_CA_PATH="$HOME/.axio/keys/AmazonRootCA1.pem"
```
Ensure directory permissions:
```bash
mkdir -p ~/.axio/keys
chmod 700 ~/.axio/keys
chmod 600 ~/.axio/keys/*
```

## Application Loading (Qt Example)
Pseudo C++ adaptation for `mqttclientcpp.cpp` (to implement later):
```cpp
QString certPath = qEnvironmentVariable("AXIO_CERT_PATH");
QString keyPath  = qEnvironmentVariable("AXIO_KEY_PATH");
QString caPath   = qEnvironmentVariable("AXIO_CA_PATH");
QSslConfiguration ssl;
ssl.setLocalCertificate(QSslCertificate::fromPath(certPath).first());
QSslKey key;
{ QFile f(keyPath); f.open(QIODevice::ReadOnly); key = QSslKey(f.readAll(), QSsl::Rsa, QSsl::Pem, QSsl::PrivateKey); }
ssl.addCaCertificates(QSslCertificate::fromPath(caPath));
```

## Rotation Procedure
1. Generate new certificate/key pair (AWS IoT or CA issuance).
2. Deploy new files to secure storage (or import into Keychain/Keystore).
3. Update environment variables (or alias labels) atomically.
4. Restart application instances.
5. Revoke / deactivate old certificate.
6. Audit logs for successful TLS handshake using new cert fingerprint.

## Auditing & Monitoring
- Log fingerprint (SHA256) of loaded certificate at startup.
- Periodic check against expected fingerprint list (config JSON).
- Raise warning if certificate expiration < 30 days.

## Incident Response
If a key leak is suspected:
1. Immediately revoke affected certificate.
2. Rotate to new credential set following rotation procedure.
3. Invalidate active sessions relying on compromised keys.
4. Review access logs (AWS IoT, server endpoints).
5. Document incident in security log and perform post‑mortem.

## Git Hygiene
- `.gitignore` excludes `/*.md` duplicates and `Keys/*` patterns.
- Keys must never re-enter version control; verify with:
```bash
git ls-files Keys/
```
(Should return empty.)

## Future Enhancements
- Integrate secret loader abstraction: `ISecretProvider` (env, keychain, keystore).
- Add automated expiration alerts via background timer.
- Support hardware secure element (TPM / Secure Enclave) for private key operations.

## Action Items
- [x] Refactor `mqttclientcpp.cpp` to load from env instead of resource path.
- [x] Add fingerprint logging.
- [x] Implement certificate expiration warning.
- [ ] Add `ISecretProvider` abstraction.

