# Camera permission test (mobile)

Page: Take attendance (attendance.html), opened over https on the same Wi-Fi

| Phone / OS | Browser | Allow camera | Block camera | Unknown face | Camera used by another app |
|---|---|---|---|---|---|
| iPhone 14 | Chrome | OK | OK | OK (attendance was not marked) | not tested yet |
| iPhone 14 | Safari | OK | OK | OK (attendance was not marked) | not tested yet |

Notes:
- Chrome: opened the full site over https on iPhone 14 (same Wi-Fi), allowed the camera, enrolled a new student and marked attendance. The "Marked Present" screen appeared.
- Chrome and Safari: blocked the camera permission and reloaded, and the camera error message was shown.
- Chrome and Safari: showed an unknown face to the camera, attendance was not marked.
- Only "camera used by another app" is still not tested.
