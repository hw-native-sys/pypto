# Security Policy

## Supported Versions

| Version / Branch | Supported |
|------------------|-----------|
| `main`           | Yes       |
| Feature branches | No        |

## Reporting a Vulnerability

**Please do not open public issues for security vulnerabilities.**

To report a vulnerability:

1. Go to the [Security Advisories](https://github.com/hw-native-sys/pypto/security/advisories) page
2. Click **"Report a vulnerability"**
3. Provide a description of the issue, steps to reproduce, and any potential impact

### What to expect

- **Acknowledgement** within 3 business days
- **Initial assessment** within 10 business days
- Regular updates until the issue is resolved or determined to be non-applicable

## Scope

The following are considered security issues:

- Arbitrary code execution via crafted IR or model inputs
- Memory safety issues in the C++ layer (buffer overflows, use-after-free)
- Vulnerabilities in third-party dependencies
- Unsafe deserialization of untrusted data

The following are generally **not** security issues:

- Crashes from intentionally malformed internal IR (developer-facing)
- Performance issues or denial-of-service from large inputs
- Issues requiring local filesystem access beyond normal usage

## Disclosure Policy

We follow coordinated disclosure:

1. Reporter submits vulnerability privately via GitHub Security Advisories
2. Team confirms, develops, and tests a fix
3. Fix is released and advisory is published
4. Reporter is credited (unless they prefer anonymity)

We ask reporters to allow a reasonable timeframe for fixes before public disclosure.
