# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | Yes |
| Previous minor | Security fixes only |
| Older versions | No |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Use [GitHub Security Advisories](https://github.com/pypto/pypto/security/advisories/new) to report vulnerabilities privately. This ensures the issue is handled confidentially until a fix is available.

Include in your report:

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact

## What to Report

- Vulnerabilities in the C++ runtime
- Issues in Python bindings (e.g., unsafe memory access across the C++/Python boundary)
- Build or CI infrastructure security issues
- Dependency vulnerabilities that affect PyPTO

## What NOT to Report

- General bugs — use [GitHub Issues](https://github.com/pypto/pypto/issues)
- Feature requests — use [GitHub Issues](https://github.com/pypto/pypto/issues)
- Questions about usage — use [GitHub Discussions](https://github.com/pypto/pypto/discussions)

## Response Timeline

| Action | Timeframe |
|--------|-----------|
| Acknowledgment | Within 3 business days |
| Status update | Within 10 business days |
| Fix development | Depends on severity and complexity |

## Disclosure Policy

PyPTO follows coordinated disclosure:

1. Vulnerability is reported privately via GitHub Security Advisories
2. The team acknowledges and triages the report
3. A fix is developed privately
4. A security advisory is published alongside the fix release
5. Credit is given to the reporter (unless they prefer anonymity)
