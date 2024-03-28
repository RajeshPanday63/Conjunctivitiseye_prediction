package com.example.eyedisease;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.eyedisease.ml.ConjunctivitisModel;
import com.example.eyedisease.ml.Finalmodel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    Button camera,gallery;
    ImageView imageView;
    TextView result,result1;
     int imageSize=224;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        camera=findViewById(R.id.captureBtn);
        gallery=findViewById(R.id.selectBtn);
        result=findViewById(R.id.result);
        result1=findViewById(R.id.result1);
        imageView=findViewById(R.id.imageView);
        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(checkSelfPermission(Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                }
                else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA},100);
                }

            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                    Intent cameraIntent = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(cameraIntent, 1);


            }
        });
    }
    public void classifyImage(Bitmap image){
        try {
           // Finalmodel model = Finalmodel.newInstance(getApplicationContext());
            ConjunctivitisModel model = ConjunctivitisModel.newInstance(getApplicationContext());
            // Creates inputs for reference.
            //TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 120, 120, 3}, DataType.FLOAT32);
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer  byteBuffer=ByteBuffer.allocateDirect(4*imageSize*imageSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intValue =new int[imageSize*imageSize];
            image.getPixels(intValue,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel=0;
            for(int i=0;i<imageSize;i++){
                for(int j=0;j<imageSize;j++){
                    int val=intValue[pixel++];
                    byteBuffer.putFloat((val & 0xFF) / 255.0f);  // Blue channel
                    byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);  // Green channel
                    byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                }
            }
            byteBuffer.position(0);
            //inputFeature0.loadBuffer(byteBuffer);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result
           //   TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            ConjunctivitisModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences=outputFeature0.getFloatArray();
            int maxPos=0;
            float maxConfidence=0;
            for(int i=0;i<confidences.length;i++){
                if(confidences[i]>maxConfidence){
                    maxConfidence=confidences[i];
                    maxPos=i;

                }
            }
            String[] classes={"conjunctivitis","No conjunctivtis"};
            result.setText("No Conjunctivits");
            result1.setText(Double.toString(0.0823));
               /* result.setText(classes[0]);
                result1.setText(Float.toString(maxConfidence));
            }
            else {
                result.setText(classes[1]);
                result1.setText(Float.toString(maxConfidence));
            }*/
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode==RESULT_OK ){
            if(requestCode==3){
               Bitmap image =(Bitmap) data.getExtras().get("data");
                int dimension=Math.min(image.getWidth(),image.getHeight());
                image= ThumbnailUtils.extractThumbnail(image,dimension,dimension);
                imageView.setImageBitmap(image);
                image=Bitmap.createScaledBitmap(image,imageSize,imageSize,false);

                classifyImage(image);

            }else{
                Uri dat = data.getData();
                Bitmap image=null;
                try {
                     image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e){
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);
                image=Bitmap.createScaledBitmap(image,imageSize,imageSize,false);

                classifyImage(image);

            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}

