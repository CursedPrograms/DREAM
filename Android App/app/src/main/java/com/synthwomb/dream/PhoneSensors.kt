package com.synthwomb.dream

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.BatteryManager
import android.os.Handler
import android.os.Looper
import org.json.JSONObject
import kotlin.math.sqrt

/**
 * The phone's sensors, lent to DREAM while the app is open: proximity (a hand or face near it),
 * the light meter, being picked up or moved, and the battery. Whatever the phone doesn't have is
 * simply left out. Readings go to her page (window.dreamPhoneSensors), which passes them on only
 * while it's her body. Sent when something changes, and every few seconds anyway.
 */
class PhoneSensors(private val context: Context, private val send: (JSONObject) -> Unit) : SensorEventListener {
    private val sm = context.getSystemService(SensorManager::class.java)
    private val proximity = sm.getDefaultSensor(Sensor.TYPE_PROXIMITY)
    private val light = sm.getDefaultSensor(Sensor.TYPE_LIGHT)
    private val motion = sm.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private val main = Handler(Looper.getMainLooper())

    private var near: Boolean? = null
    private var lux: Float? = null
    private var moved = false
    private var sentNear: Boolean? = null
    private var sentLux = -1f
    private var lastSent = 0L

    private val tick = object : Runnable {
        override fun run() {
            report(force = false)
            main.postDelayed(this, 1000)
        }
    }

    private var running = false

    fun start() {
        if (running) return
        running = true
        listOfNotNull(proximity, light, motion).forEach { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
        main.post(tick)
    }

    fun stop() {
        running = false
        sm.unregisterListener(this)
        main.removeCallbacks(tick)
    }

    override fun onSensorChanged(e: SensorEvent) {
        when (e.sensor.type) {
            Sensor.TYPE_PROXIMITY -> near = e.values[0] < minOf(e.sensor.maximumRange, 5f)
            Sensor.TYPE_LIGHT -> lux = e.values[0]
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                val (x, y, z) = Triple(e.values[0], e.values[1], e.values[2])
                if (sqrt(x * x + y * y + z * z) > 2.5f) moved = true     // picked up or bumped, not a hum
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun report(force: Boolean) {
        val now = System.currentTimeMillis()
        val l = lux
        val changed = near != sentNear || moved ||
            (l != null && (sentLux < 0 || kotlin.math.abs(l - sentLux) > maxOf(5f, sentLux * 0.3f)))
        if (!force && !changed && now - lastSent < 5000) return
        val r = JSONObject()
        near?.let { r.put("near", it) }
        l?.let { r.put("lux", it.toDouble()) }
        r.put("moved", moved)
        battery(r)
        send(r)
        sentNear = near
        if (l != null) sentLux = l
        moved = false
        lastSent = now
    }

    private fun battery(r: JSONObject) {
        val b = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return
        val level = b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = b.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        if (level >= 0 && scale > 0) r.put("battery", level * 100.0 / scale)
        r.put("charging", b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0)
    }
}
