package com.example.pcd

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import com.example.pcd.databinding.ActivityMainBinding
import com.example.pcd.ml.Durian.newInstance


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var mBitmap: Bitmap
    private val imageSize = 225

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding =  ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnGaleri.setOnClickListener {
            val callGalleryIntent = Intent(Intent.ACTION_PICK)
            callGalleryIntent.type = "image/*"
            startActivityForResult(callGalleryIntent, 2)
        }
    }

    private fun classifyImage(image: Bitmap) {
        try {
            val model = newInstance(applicationContext)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 225, 225, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(imageSize * imageSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
            var pixel = 0
            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val `val` = intValues[pixel++]
                    byteBuffer.putFloat(((`val` shr 16) and 0xFF) * (1f / 255f))
                    byteBuffer.putFloat(((`val` shr 8) and 0xFF) * (1f / 255f))
                    byteBuffer.putFloat((`val` and 0xFF) * (1f / 255f))
                }
            }

            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            val confidences = outputFeature0.floatArray
            var maxPos = 0
            var maxConfidence = 0f
            for (i in confidences.indices) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i]
                    maxPos = i
                }
            }

            val classes = arrayOf("Musang King", "Bawor", "Super Tembaga", "Duri Hitam")

            if (maxConfidence < 0.7f) {
                binding.result.text = "Bukan gambar daun durian"
                binding.result.setTextColor(ContextCompat.getColor(this, android.R.color.holo_red_dark))

                binding.confidence.text = ""
            } else {
                binding.result.text = classes[maxPos]
                binding.result.setTextColor(ContextCompat.getColor(this, R.color.black))

                var s = ""
                for (i in classes.indices) {
                    s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100)
                }
                binding.confidence.text = s
            }

            model.close()
        } catch (e: IOException) {
            Toast.makeText(this, "Gagal memproses gambar.", Toast.LENGTH_SHORT).show()
            e.printStackTrace()
        }
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {

        if (requestCode == 1) {

            if (resultCode == RESULT_OK && data != null) {

                mBitmap = data.extras!!.get("data") as Bitmap
                val dimension = mBitmap.width.coerceAtMost(mBitmap.height)
                val thumbnail = ThumbnailUtils.extractThumbnail(mBitmap, dimension, dimension)
                binding.imageView.setImageBitmap(thumbnail)

                val scaledImage = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
                classifyImage(scaledImage)
            } else {
                Toast.makeText(this, "Camera cancel..", Toast.LENGTH_LONG).show()
            }
        } else if (requestCode == 2) {
            if (data != null) {
                val uri = data.data

                try {
                    mBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                } catch (e: IOException) {
                    e.printStackTrace()
                }

                println("Success!!!")
                val dimension = mBitmap.width.coerceAtMost(mBitmap.height)
                val thumbnail = ThumbnailUtils.extractThumbnail(mBitmap, dimension, dimension)
                binding.imageView.setImageBitmap(thumbnail)

                val scaledImage = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
                classifyImage(scaledImage)

            }
        } else {
            Toast.makeText(this, "Unrecognized request code", Toast.LENGTH_LONG).show()

        }
        super.onActivityResult(requestCode, resultCode, data)
    }
}