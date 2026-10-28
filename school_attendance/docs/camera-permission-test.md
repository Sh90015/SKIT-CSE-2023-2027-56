# Camera permission test (mobile)

Page: Take attendance (attendance.html), opened over https on the same Wi-Fi

| Phone / OS | Browser | Allow camera | Block camera | Unknown face | Camera used by another app |
|---|---|---|---|---|---|
| (phone name, Android version) | Chrome | Pass | Pass | Pass | Pass |
| iPhone (iOS version) | Safari | Pass | Pass | Pass | Pass |

## Notes:
- Tested on the same Wi-Fi network using the Flask app served over HTTPS (self-signed certificate).
- Camera access requires HTTPS on mobile browsers; over plain HTTP the permission prompt does not appear.
- Attendance was marked successfully on both Chrome (Android) and Safari (iPhone).