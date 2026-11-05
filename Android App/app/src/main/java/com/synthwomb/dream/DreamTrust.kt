package com.synthwomb.dream

import android.content.SharedPreferences
import android.net.http.SslCertificate
import android.os.Build
import java.security.MessageDigest

/**
 * DREAM's page is HTTPS with a certificate her PC made itself (app.py), so Android can't vouch
 * for it. Instead of skipping the check, the app asks once ("is this her?") and remembers the
 * certificate's SHA-256 for that address. Later visits go through only with that exact certificate.
 * app.py prints the same fingerprint at start-up, so you can compare. A new certificate (her PC
 * got a new address) asks again.
 */
class DreamTrust(private val prefs: SharedPreferences) {

    fun pinned(host: String): String? = prefs.getString("cert_$host", null)

    fun pin(host: String, fingerprint: String) = prefs.edit().putString("cert_$host", fingerprint).apply()

    fun forget(host: String) = prefs.edit().remove("cert_$host").apply()

    companion object {
        /** "AB:CD:..." SHA-256 of the certificate, the format app.py prints; null if unreadable. */
        fun fingerprint(cert: SslCertificate?): String? {
            cert ?: return null
            val der: ByteArray = (if (Build.VERSION.SDK_INT >= 29) cert.x509Certificate?.encoded
                else SslCertificate.saveState(cert).getByteArray("x509-certificate")) ?: return null
            return hex(MessageDigest.getInstance("SHA-256").digest(der))
        }

        fun hex(bytes: ByteArray): String = bytes.joinToString(":") { "%02X".format(it) }
    }
}
