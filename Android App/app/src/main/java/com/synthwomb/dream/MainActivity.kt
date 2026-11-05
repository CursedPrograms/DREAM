package com.synthwomb.dream

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.Uri
import android.net.http.SslError
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.webkit.PermissionRequest
import android.webkit.SslErrorHandler
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

private const val NAME = "DREAM"
private const val PORT = 5001                       // her phone site (app.py PhonePort), HTTPS
private const val PATH = "/dream.html"
private const val SERVICE = "_flask-link._tcp."     // what app.py announces on the network
private val BG = Color.parseColor("#070A12")
private val PANEL = Color.parseColor("#0D1221")
private val EDGE = Color.parseColor("#192543")
private val TEXT = Color.parseColor("#E9ECF2")
private val ACCENT = Color.parseColor("#4C8DFF")

/**
 * DREAM on the phone: her own page from the PC that runs her (app.py), full screen.
 * The page keeps the rules of who she is with: the newest open page of hers is her body
 * (shows, speaks, listens), and with a camera it's her eyes too. Opening the app moves her
 * to the phone; closing it or switching away gives her back to the PC within seconds.
 * Turning the phone swaps her clips between landscape and portrait.
 */
class MainActivity : Activity() {
    private lateinit var web: WebView
    private lateinit var panel: LinearLayout
    private lateinit var address: EditText
    private lateinit var state: TextView
    private lateinit var trust: DreamTrust
    private lateinit var sensors: PhoneSensors
    private val prefs by lazy { getPreferences(Context.MODE_PRIVATE) }
    private var host = ""
    private var port = PORT
    private var pendingWeb: PermissionRequest? = null
    private var wifi: ConnectivityManager.NetworkCallback? = null
    private var wasAway = false
    private var asking = false               // our own permission dialog pauses us too: not "away"

    private fun button(text: String, action: () -> Unit) = Button(this).apply {
        this.text = text
        setTextColor(TEXT)
        setBackgroundColor(PANEL)
        setOnClickListener { action() }
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(saved: Bundle?) {
        super.onCreate(saved)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        trust = DreamTrust(prefs)
        // The phone's sensors go to her page; it passes them on only while it's her body
        sensors = PhoneSensors(this) { r ->
            if (web.visibility == View.VISIBLE) web.evaluateJavascript("window.dreamPhoneSensors && window.dreamPhoneSensors($r)", null)
        }

        web = WebView(this).apply {
            setBackgroundColor(Color.BLACK)
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.mediaPlaybackRequiresUserGesture = false
            webViewClient = DreamClient()
            webChromeClient = object : WebChromeClient() {
                override fun onPermissionRequest(request: PermissionRequest) = runOnUiThread { allow(request) }
            }
            visibility = View.GONE
        }

        state = TextView(this).apply { setTextColor(TEXT); gravity = Gravity.CENTER; setPadding(0, 24, 0, 24) }
        address = EditText(this).apply {
            hint = "her PC's address"
            setText(prefs.getString("address", "") ?: "")
            setTextColor(TEXT)
            setHintTextColor(EDGE)
            setSingleLine()
        }
        panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(BG)
            setPadding(48, 48, 48, 48)
            addView(TextView(context).apply { text = NAME; textSize = 32f; setTextColor(ACCENT); gravity = Gravity.CENTER })
            addView(TextView(context).apply {
                text = "Opens DREAM from the PC she runs on. While this app is open she is with you on the phone."
                setTextColor(TEXT); gravity = Gravity.CENTER; setPadding(0, 16, 0, 24)
            })
            addView(address, LinearLayout.LayoutParams(-1, -2))
            addView(LinearLayout(context).apply {
                gravity = Gravity.CENTER
                addView(button("FIND") { find() }, LinearLayout.LayoutParams(0, -2, 1f))
                addView(button("CONNECT") { connect(address.text.toString()) }, LinearLayout.LayoutParams(0, -2, 1f))
            }, LinearLayout.LayoutParams(-1, -2))
            addView(state)
        }

        setContentView(FrameLayout(this).apply {
            setBackgroundColor(BG)
            addView(panel, FrameLayout.LayoutParams(-1, -1))
            addView(web, FrameLayout.LayoutParams(-1, -1))
        })
        showSplash()
        askForEarsAndEyes()

        val last = prefs.getString("address", "") ?: ""
        if (last.isNotBlank()) connect(last) else say("Press FIND, or type her PC's address")
    }

    private fun say(text: String) = runOnUiThread { state.text = text }

    // ---- her ears and eyes: Android's permission first, then the page's request ----
    private val needed = arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA)

    private fun askForEarsAndEyes() {
        val missing = needed.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) { asking = true; requestPermissions(missing.toTypedArray(), 1) }
    }

    override fun onRequestPermissionsResult(code: Int, permissions: Array<out String>, results: IntArray) {
        asking = false
        pendingWeb?.let { pendingWeb = null; allow(it) }
    }

    /** Only her page gets the mic and camera, and only what Android allowed. */
    private fun allow(request: PermissionRequest) {
        if (request.origin.host != host) return request.deny()
        val want = request.resources.filter {
            (it == PermissionRequest.RESOURCE_AUDIO_CAPTURE && has(Manifest.permission.RECORD_AUDIO)) ||
                (it == PermissionRequest.RESOURCE_VIDEO_CAPTURE && has(Manifest.permission.CAMERA))
        }
        val missing = needed.filter { !has(it) }
        if (want.size < request.resources.size && missing.isNotEmpty() && pendingWeb == null) {
            pendingWeb = request                  // ask Android, then answer the page
            asking = true
            requestPermissions(missing.toTypedArray(), 2)
            return
        }
        if (want.isEmpty()) request.deny() else request.grant(want.toTypedArray())
    }

    private fun has(p: String) = checkSelfPermission(p) == PackageManager.PERMISSION_GRANTED

    // ---- connect: "192.168.0.100" or "192.168.0.100:5001" ----
    private fun connect(typed: String) {
        val text = typed.trim().removePrefix("https://").removePrefix("http://").substringBefore('/')
        if (text.isEmpty()) return say("Type her PC's address, or press FIND")
        host = text.substringBefore(':')
        port = text.substringAfter(':', "").toIntOrNull() ?: PORT
        prefs.edit().putString("address", text).apply()
        runOnUiThread {
            address.setText(text)
            state.text = "Connecting to $host..."
            web.visibility = View.VISIBLE
            web.loadUrl("https://$host:$port$PATH?autostart=1")   // opening the app means "come here"
        }
    }

    private fun backToPanel(why: String) = runOnUiThread {
        web.loadUrl("about:blank")              // her page's pagehide: she goes back to the PC
        web.visibility = View.GONE
        state.text = why
    }

    private inner class DreamClient : WebViewClient() {
        override fun onReceivedSslError(view: WebView, handler: SslErrorHandler, error: SslError) {
            val at = Uri.parse(error.url).host
            val fp = DreamTrust.fingerprint(error.certificate)
            // Her own certificate is "untrusted" (no authority signed it). Anything else is a real fault.
            if (at != host || fp == null || error.primaryError != SslError.SSL_UNTRUSTED) {
                handler.cancel()
                return backToPanel("$host's certificate is wrong (error ${error.primaryError}). Is this her PC?")
            }
            if (trust.pinned(host) == fp) return handler.proceed()
            val changed = trust.pinned(host) != null
            AlertDialog.Builder(this@MainActivity)
                .setTitle(if (changed) "DREAM's certificate changed" else "Is this DREAM?")
                .setMessage(
                    (if (changed) "$host shows a different certificate than last time. That's normal after her PC " +
                        "got a new address or a new certificate; otherwise something else is answering.\n\n"
                    else "Her PC made its own certificate, so the phone can't check it by itself.\n\n") +
                        "SHA-256:\n$fp\n\nWhen DREAM starts, her PC prints \"Certificate SHA-256\". Trust it if they match."
                )
                .setPositiveButton("Trust") { _, _ -> trust.pin(host, fp); handler.proceed() }
                .setNegativeButton("Cancel") { _, _ -> handler.cancel(); backToPanel("Not connected") }
                .setCancelable(false)
                .show()
        }

        override fun onReceivedError(view: WebView, request: WebResourceRequest, error: android.webkit.WebResourceError) {
            if (request.isForMainFrame) backToPanel("Can't reach DREAM at $host:$port. Is she running?")
        }
    }

    // ---- FIND: RIFT's fleet list (as the robots' apps do), else her own announcement ----
    private fun find() = thread(isDaemon = true) {
        say("Looking for DREAM...")
        val ip = fromFleet() ?: fromNetwork() ?: return@thread say("DREAM isn't on this network. Is she running?")
        say("Found her at $ip")
        connect(ip)
    }

    private fun fromFleet(): String? = try {
        val wm = applicationContext.getSystemService(WifiManager::class.java)
        @Suppress("DEPRECATION") val g = wm.dhcpInfo.gateway
        val gateway = "${g and 255}.${g shr 8 and 255}.${g shr 16 and 255}.${g shr 24 and 255}"
        val body = listOf("/fleet", "/robots").firstNotNullOfOrNull { get("http://$gateway:5000$it") }
        val robots = JSONObject(body ?: "{}").optJSONArray("robots")
        (0 until (robots?.length() ?: 0)).map { robots!!.getJSONObject(it) }
            .firstOrNull { it.optString("name").equals(NAME, ignoreCase = true) }
            ?.optString("ip")?.takeIf { it.isNotBlank() }
    } catch (_: Exception) {
        null
    }

    private fun get(url: String): String? = try {
        val c = (URL(url).openConnection() as HttpURLConnection).apply { connectTimeout = 1500; readTimeout = 2000 }
        try { c.inputStream.bufferedReader().readText() } finally { c.disconnect() }
    } catch (_: Exception) {
        null
    }

    /** app.py announces "DREAM" as _flask-link._tcp on the network. */
    @Suppress("DEPRECATION")
    private fun fromNetwork(): String? {
        val nsd = getSystemService(NsdManager::class.java)
        val done = CountDownLatch(1)
        var found: String? = null
        val listener = object : NsdManager.DiscoveryListener {
            override fun onServiceFound(info: NsdServiceInfo) {
                if (!info.serviceName.equals(NAME, ignoreCase = true)) return
                nsd.resolveService(info, object : NsdManager.ResolveListener {
                    override fun onServiceResolved(r: NsdServiceInfo) { found = r.host?.hostAddress; done.countDown() }
                    override fun onResolveFailed(r: NsdServiceInfo, code: Int) {}
                })
            }
            override fun onDiscoveryStarted(type: String) {}
            override fun onDiscoveryStopped(type: String) {}
            override fun onServiceLost(info: NsdServiceInfo) {}
            override fun onStartDiscoveryFailed(type: String, code: Int) { done.countDown() }
            override fun onStopDiscoveryFailed(type: String, code: Int) {}
        }
        val lock = applicationContext.getSystemService(WifiManager::class.java).createMulticastLock("dream-find")
        lock.acquire()
        try {
            nsd.discoverServices(SERVICE, NsdManager.PROTOCOL_DNS_SD, listener)
            done.await(8, TimeUnit.SECONDS)
            try { nsd.stopServiceDiscovery(listener) } catch (_: Exception) {}
        } finally {
            lock.release()
        }
        return found
    }

    // ---- she's with the phone only while the app is in front ----
    override fun onResume() {
        super.onResume()
        web.resumeTimers()
        web.onResume()
        sensors.start()
        // Back from another app: a fresh page is the newest of hers again, so she comes back here
        if (wasAway && web.visibility == View.VISIBLE) web.reload()
        wasAway = false
        // Her PC's network may have no internet: keep talking over WiFi, not mobile data
        val cm = getSystemService(ConnectivityManager::class.java)
        val cb = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) { cm.bindProcessToNetwork(network) }
        }
        wifi = cb
        cm.requestNetwork(NetworkRequest.Builder().addTransportType(NetworkCapabilities.TRANSPORT_WIFI).build(), cb)
    }

    override fun onPause() {
        if (!asking) {
            // Her page says goodbye (the same as closing its tab), so she's back on the PC at once
            web.evaluateJavascript("window.dispatchEvent(new Event('pagehide'))", null)
            web.onPause()
            web.pauseTimers()
            sensors.stop()
            wasAway = true
        }
        wifi?.let { getSystemService(ConnectivityManager::class.java).unregisterNetworkCallback(it) }
        wifi = null
        super.onPause()
    }

    override fun onDestroy() {
        web.loadUrl("about:blank")
        web.destroy()
        super.onDestroy()
    }

    @Deprecated("Back from her page shows the connect screen; back again closes the app")
    override fun onBackPressed() {
        if (web.visibility == View.VISIBLE) backToPanel("Disconnected") else super.onBackPressed()
    }
}
