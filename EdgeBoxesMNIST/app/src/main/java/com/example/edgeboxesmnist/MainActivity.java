package com.example.edgeboxesmnist;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.support.annotation.Nullable;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import org.opencv.imgproc.Imgproc;
public class MainActivity extends AppCompatActivity {
    ImageView ivDigit;
    Button btnCamera;
    Button btnPredict;
    TextView tvDebug;
    Interpreter tflite;
    Interpreter.Options options = new Interpreter.Options();
    private static final int IMAGE_PICK_CODE = 1000;
    ByteBuffer imgData = null;
    int[] imgPixels = new int[28*28];
    float[][] result = new float[1][10];
    double[][] gaussianKernel ={{0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902},
            {0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621},
            {0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823},
            {0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621},
            {0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902}};
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ivDigit = findViewById(R.id.ivImage);
        btnCamera = findViewById(R.id.btnCamera);
        btnPredict = findViewById(R.id.btnPredict);
        tvDebug = findViewById(R.id.tvDebug);
        try {
            tflite = new Interpreter(loadModelFile(), options);
        } catch (Exception e) {
            e.printStackTrace();
        }
        imgData = ByteBuffer.allocateDirect(4*28*28);
        imgData.order(ByteOrder.nativeOrder());
        btnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                chooseImageFromGallery();
            }
        });
        btnPredict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ivDigit.invalidate();
                BitmapDrawable drawable = (BitmapDrawable) ivDigit.getDrawable();
                Bitmap bitmap = drawable.getBitmap();
                OpenCVLoader.initDebug();
                Mat imageMat = new Mat();
                Utils.bitmapToMat(bitmap, imageMat);
                Mat grayImageMat = new Mat();
                Imgproc.cvtColor(imageMat, grayImageMat, Imgproc.COLOR_BGR2GRAY);
                Imgproc.threshold(grayImageMat, grayImageMat, 100, 255, Imgproc.THRESH_BINARY);
                Mat blurredMat = new Mat();
                double[][] img = matToArray(blurredMat);
                double[][] blurredImage = convolve(img, gaussianKernel);
                double[][][]  edgeImg_theta = sobelFilters(blurredImage);
                double[][] thinEdge = nonMaxSuppression(edgeImg_theta);
                double[][] thresholdedImg = threshold(thinEdge, 0.05, 0.09);
                double[][] edgedImg = hysteresis(thresholdedImg, 25, 255);
                blurredMat = convertMat(edgedImg);
                Imgproc.GaussianBlur(grayImageMat, blurredMat, new Size(13, 13), 13);
                Mat mHierarchy = new Mat();
                List<MatOfPoint> contours = new ArrayList<>();
                Imgproc.findContours(blurredMat, contours, mHierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
//                Mat drawing = Mat.zeros(blurredMat.size(), CvType.CV_8UC3);
//                Imgproc.drawContours(drawing, contours, -1, new Scalar(255, 0, 0), 7);
                String labels = "";
                Mat boxes = imageMat.clone();
                for (MatOfPoint contour: contours) {
                    if (Imgproc.contourArea(contour) < 500)
                        continue;
                    Rect boundingBox = Imgproc.boundingRect(contour);
                    if (boundingBox.x > 50 && boundingBox.y > 50 && boundingBox.x+boundingBox.width+100 < grayImageMat.width() && boundingBox.y+boundingBox.height+100 < grayImageMat.height()) {
                        boundingBox.x -= 50;
                        boundingBox.y -= 50;
                        boundingBox.width += 100;
                        boundingBox.height += 100;
                        if (boundingBox.width > boundingBox.height) {
                            boundingBox.y = boundingBox.y - (boundingBox.width- boundingBox.height)/2;
                            boundingBox.height = boundingBox.width;
                        }
                        else {
                            boundingBox.x = boundingBox.x - (boundingBox.height- boundingBox.width)/2;
                            boundingBox.width = boundingBox.height;
                        }
                        Imgproc.rectangle(boxes, boundingBox, new Scalar(0, 255, 0), 10);
                        Bitmap bitmapImage = Bitmap.createBitmap(grayImageMat.cols(), grayImageMat.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(grayImageMat, bitmapImage);
                        Bitmap croppedImage = Bitmap.createBitmap(bitmapImage, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height);
                        Bitmap bitmap_resize = getResizedBitmap(croppedImage, 28, 28);
                        convertBitmapToByteBuffer(bitmap_resize);
                        tflite.run(imgData, result);

                        if (!labels.contains(Integer.toString(argmax(result[0]))))
                            labels = labels + Integer.toString(argmax(result[0])) + " ";
                        }
                }
                String nums[] = labels.split(" ");
                String result = "The digits that were detect in the image are: ";
                for (int i=0; i<nums.length; i++) {
                    if (i == nums.length-1) {
                        result += nums[i] +".";
                        break;
                    }
                    result += nums[i]+", ";
                }
                tvDebug.setText(result);
                Bitmap boundedImage = Bitmap.createBitmap(boxes.cols(), boxes.rows(), Bitmap.Config.RGB_565);
                Utils.matToBitmap(boxes, boundedImage);
                ivDigit.setImageBitmap(boundedImage);
            }
        });
    }
    public static int[][] reshape(int[][] origImg){
        int[][] newImg = new int[28][28];
        int origWidth = origImg.length;
        int origHeight = origImg[0].length;
        double widthScale = origWidth/28;
        double heightScale = origHeight/28;
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                double xIdx = i * widthScale;
                double yIdx = j * heightScale;
                int xFloor = (int)Math.floor(xIdx);
                if(xFloor < 0){
                    xFloor = 0;
                }
                int yFloor = (int)Math.floor(yIdx);
                if(yFloor < 0){
                    yFloor = 0;
                }
                int xCeil = (int)Math.ceil(xIdx);
                if(xCeil > origWidth){
                    xCeil = origWidth;
                }
                int yCeil = (int)Math.ceil(yIdx);
                if(yCeil > origHeight){
                    yCeil = origHeight;
                }
                int neigh1 = origImg[xFloor][yFloor];
                int neigh2 = origImg[xFloor][yCeil];
                int neigh3 = origImg[xCeil][yFloor];
                int neigh4 = origImg[xCeil][yCeil];
                double dist1 = Math.sqrt((xFloor-xIdx)*(xFloor-xIdx) + (yFloor-yIdx)*(yFloor-yIdx));
                double dist2 = Math.sqrt((xFloor-xIdx)*(xFloor-xIdx) + (yCeil-yIdx)*(yCeil-yIdx));
                double dist3 = Math.sqrt((xCeil-xIdx)*(xCeil-xIdx) + (yFloor-yIdx)*(yFloor-yIdx));
                double dist4 = Math.sqrt((xCeil-xIdx)*(xCeil-xIdx) + (yCeil-yIdx)*(yCeil-yIdx));
                double total = dist1+dist2+dist3+dist4;
                if(total == 0){
                    newImg[i][j] = Math.round(neigh1/4 + neigh2/4 + neigh3/4 + neigh4/4);
                } else {
                    newImg[i][j] = (int)Math.round(neigh1*(1-(dist1/total)) + neigh2*(1-(dist2/total)) +
                            neigh3*(1-(dist3/total)) + neigh4*(1-(dist4/total)));
                }
            }
        }
        return newImg;
    }
    public static double[][] matToArray(Mat img) {
        double[][] img_ = new double[img.rows()][img.cols()];
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                double[] data = img.get(i, j);
                img_[i][j] = data[0];
            }
        }
        return img_;
    }
    private Mat convertMat(double[][] img) {
        int rows = img.length;
        int cols = img[0].length;
        Mat mat = new Mat(rows, cols, CvType.CV_32SC1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat.put(i, j, img[i][j]);
            }
        }
        return mat;
    }
    private double[][] getImageAsArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        double[][] grayArray = new double[height][width];
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int pixel = pixels[i * width + j];
                int red = Color.red(pixel);
                int green = Color.green(pixel);
                int blue = Color.blue(pixel);
                double gray = red*.299 + green*.587 + blue*.114;
                grayArray[i][j] = gray;
            }
        }
        return grayArray;
    }
    private int argmax(float[] probs) {
        int maxIds = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIds = i;
            }
        }
        return maxIds;
    }
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(imgPixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                int value = imgPixels[pixel++];
                imgData.putFloat(convertPixel(value));
            }
        }
    }
    private static float convertPixel(int color) {
        return (255-(((color>>16) & 0xFF)*0.299f
                +((color>>8) & 0xFF)*0.587f
                +(color & 0xFF)*0.114f))/255.0f;
    }
    private Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float)newWidth)/width;
        float scaleHeight = ((float)newHeight)/height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        return Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false);
    }
        private Bitmap getResizebidBitmap(Bitmap bm, int newWidth, int newHeight) {
            Matrix matrix = new Matrix();
            matrix.postScale((float)newWidth / bm.getWidth(), (float)newHeight / bm.getHeight());
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bm, newWidth, newHeight, true);
            return resizedBitmap;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == IMAGE_PICK_CODE) {
            assert data != null;
            ivDigit.setImageURI(data.getData());
        }
    }
    private void chooseImageFromGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, IMAGE_PICK_CODE);
    }
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    public static double[][][] sobelFilters(double[][] img) {
        double[][][]  edgeImg_theta = new double[img.length][img[0].length][2];
        double[][] Kx = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        double[][] Ky = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
        double[][] Ix = convolve(img, Kx);
        double[][] Iy = convolve(img, Ky);
        for (int i = 0; i < edgeImg_theta.length; i++){
            for (int j = 0; j < edgeImg_theta[0].length; j++){
                edgeImg_theta[i][j][0] = Math.sqrt(Math.pow(Ix[i][j], Ix[i][j]) +Math.pow(Iy[i][j], Iy[i][j]));
                edgeImg_theta[i][j][1] = Math.atan2(Iy[i][j], Ix[i][j]);
            }
        }
        double max_val = edgeImg_theta[0][0][0];
        for (int i = 0; i < edgeImg_theta.length; i++){
            for (int j = 0; j < edgeImg_theta[0].length; j++){
                if (edgeImg_theta[i][j][0] > max_val){
                    max_val = edgeImg_theta[i][j][0];
                }
            }
        }
        for (int i = 0; i < edgeImg_theta.length; i++){
            for (int j = 0; j < edgeImg_theta[0].length; j++){
                edgeImg_theta[i][j][0] = edgeImg_theta[i][j][0] * max_val / 255;
            }
        }
        return edgeImg_theta;
    }
    public static double[][] nonMaxSuppression(double[][][] img){
        double[][] thin_edge = new double[img.length][img[0].length];
        double[][] angle = new double[img.length][img[0].length];
        for (int i = 0; i < img.length; i++){
            for (int j = 0; j < img[0].length; j++){
                angle[i][j] = img[i][j][1] * 180 / Math.PI;
                if (angle[i][j] < 0){
                    angle[i][j] += 180;
                }
            }
        }
        for (int i = 1; i < img.length; i++){
            for (int j = 1; j < img[0].length; j++){
                try {
                    double q = 255;
                    double r = 255;
                    if ((0 <= angle[i][j] &&  angle[i][j] < 22.5) || (157.5 <= angle[i][j] && angle[i][j] <= 180)){
                        q = img[i][j+1][0];
                        r = img[i][j-1][0];
                    }
                    else if (22.5 <= angle[i][j] && angle[i][j] < 67.5){
                        q = img[i+1][j-1][0];
                        r = img[i-1][j+1][0];
                    }
                    else if (67.5 <= angle[i][j] && angle[i][j] < 112.5){
                        q = img[i+1][j][0];
                        r = img[i-1][j][0];
                    }
                    else if (112.5 <= angle[i][j] && angle[i][j] < 157.5){
                        q = img[i-1][j-1][0];
                        r = img[i+1][j+1][0];
                    }
                    if ((img[i][j][0] >= q) && (img[i][j][0] >= r))
                        thin_edge[i][j] = (int)Math.floor(img[i][j][0]);
                    else
                        thin_edge[i][j] = 0;
                } catch(Exception e) {
                    ;
                }
            }
        }
        return thin_edge;
    }
    public static double[][] threshold(double[][]img, double lowThresholdRatio, double highThresholdRatio){ //lTR = 0.05, hTR = 0.09
        double[][] thresholdedImg = new double[img.length][img[0].length];
        int max_val = (int) img[0][0];
        for (int i = 0; i < img.length; i++){
            for (int j = 0; j < img[0].length; j++){
                if (img[i][j] > max_val){
                    max_val = (int) img[i][j];
                }
            }
        }
        double highThreshold = max_val * highThresholdRatio;
        double lowThreshold = highThreshold * lowThresholdRatio;
        int weak = 25;
        int strong = 255;
        for (int i = 0; i < img.length; i++){
            for (int j = 0; j < img[0].length; j++){
                if ((img[i][j] <= highThreshold) & (img[i][j] >= lowThreshold)) {
                    thresholdedImg[i][j] = weak;
                }
                else if (img[i][j] >= highThreshold){
                    thresholdedImg[i][j] = strong;
                }
                else {
                    thresholdedImg[i][j] = 0;
                }
            }
        }
        return thresholdedImg;
    }
    public static double[][] hysteresis(double[][]img, int weak, int strong){ //w = 25, s = 225
        for (int i = 1; i < img.length; i++){
            for (int j = 1; j < img[0].length; j++){
                if (img[i][j] == weak){
                    try {
                        if ((img[i+1][j-1] == strong) || (img[i+1][j] == strong) || (img[i+1][j+1] == strong)
                                || (img[i][j-1] == strong) || (img[i][j+1] == strong)
                                || (img[i-1][j-1] == strong) || (img[i-1][j] == strong) || (img[i-1][j+1] == strong)){
                            img[i][j] = strong;
                        }
                    } catch(Exception e) {
                        ;
                    }
                }
            }
        }
        return img;
    }
    public static double[][] convolve(double[][] pic, double[][] kernel) {
        if (pic.length == 0)
            return new double[1][1];
        double[][] picConv = new double[pic.length][pic[0].length];
        double[][] kernelFlip = new double[kernel.length][kernel[0].length];
        for (int i = 0; i < kernel.length; i++) {
            for (int j = 0; j < kernel[i].length; j++) {
                kernelFlip[i][j] = kernel[kernel.length-1-i][kernel[0].length-1-j];
            }
        }
        for (int i = 0; i < pic.length; i++) {
            for (int j = 0; j < pic[i].length; j++) {
                double sumR = 0;
                for (int m = 0; m < kernelFlip.length; m++) {
                    for (int n = 0; n < kernelFlip[m].length; n++) {
                        int mIndex = (int) (-(kernelFlip.length-1)/2 + m);
                        int nIndex = (int) (-(kernelFlip[0].length-1)/2 + n);
                        if (i+mIndex < 0 || j+nIndex < 0 || i+mIndex >= pic.length || j+nIndex >= pic[i].length) {
                            continue;
                        }
                        sumR += pic[i+mIndex][j+nIndex] * kernelFlip[m][n];
                    }
                }
                picConv[i][j] = sumR;
            }
        }
        return picConv;
    }
}