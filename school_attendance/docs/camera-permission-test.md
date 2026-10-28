# Camera Permission Test (Mobile)

**Page:** Take attendance (`attendance.html`)  
**Environment:** Opened over HTTPS on the same Wi-Fi network.

## Detailed test results

| Phone / OS | Browser | Allow camera | Block camera | Unknown face | Camera used by another app |
|---|---|---|---|---|---|
| iPhone 14 | Chrome | OK | OK | OK (attendance was not marked) | Not tested yet |
| iPhone 14 | Safari | OK | OK | OK (attendance was not marked) | Not tested yet |

## Notes

- **Chrome on iPhone 14:** Opened the full site over HTTPS on the same Wi-Fi, allowed camera access, enrolled a new student, and marked attendance. The “Marked Present” screen appeared.
- **Chrome and Safari:** When camera permission was blocked and the page was reloaded, the camera error message was shown.
- **Chrome and Safari:** When an unknown face was shown to the camera, attendance was not marked.
- **Not tested yet:** The “camera used by another app” scenario.
- The Flask app was served over HTTPS using a self-signed certificate.
- Camera access requires HTTPS on mobile browsers; over plain HTTP, the permission prompt may not appear.

## Additional previously recorded results

The earlier notes also recorded the following results, but did not specify exact device models or OS versions:

| Phone / OS | Browser | Allow camera | Block camera | Unknown face | Camera used by another app |
|---|---|---|---|---|---|

| Android (model/version not specified) | Chrome | Pass | Pass | Pass | Pass |
| iPhone (iOS version not specified) | Safari | Pass | Pass | Pass | Pass |

**Note:** The additional results above are kept separately because their device details and test conditions were not fully specified. Confirm them before treating them as equivalent to the detailed iPhone 14 tests.

| (phone name, Android version) | Chrome | Pass | Pass | Pass | Pass |
| iPhone (iOS version) | Safari | Pass | Pass | Pass | Pass |

## Notes:
- Tested on the same Wi-Fi network using the Flask app served over HTTPS (self-signed certificate).
- Camera access requires HTTPS on mobile browsers; over plain HTTP the permission prompt does not appear.
- Attendance was marked successfully on both Chrome (Android) and Safari (iPhone).

