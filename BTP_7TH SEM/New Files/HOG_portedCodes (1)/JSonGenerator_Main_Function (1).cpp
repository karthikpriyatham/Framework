  #include<bits/stdc++.h>
  #include"HOGImage.h"

  using namespace std;
  using namespace HOG;

  /*
    
    Enter these in terminal 
    Executing commands :: 1)  g++ JSonGenerator.cpp HOGImage.cpp HOGImage.h -Wno-write-strings  -lfreeimage
                          2)  clear && ./a.out > Output.txt


    If the above 2 commands return an error such as install freeimage.h , you can do it via
    First Install Free Image library via :
        sudo apt-get install libfreeimage-dev and then try 1 and 2 commands in ubuntu terminal 
        or bash 



  */

  
  #define convKernelRadius 1
  #define convKernelWidth (2 * convKernelRadius + 1)
  #define convRowTileWidth 128
  #define convKernelRadiusAligned 16

  #define convColumnTileWidth 16
  #define convColumnTileHeight 48

  const int convKernelSize = convKernelWidth * sizeof(float); //3*float
  int gblworksize,partit,workdimension,id=1;
  char name[100],src[100],ip[100];
  char type[6][30]={"uchar","uint","char","int","long","float"};
  int hPaddedWidth,hPaddedHeight,hPaddingSizeX,hPaddingSizeY;
  int toaddxx, toaddxy, toaddyx, toaddyy,avSizeX,avSizeY,marginX,marginY;
  bool hUseGrayscale;
  int hNoHistogramBins,hcellSizeX,hcellSizeY,hblockSizeX,hblockSizeY,hwindowSizeX,hwindowSizeY,hNoOfCellsX,hNoOfCellsY;
  int hNoOfBlocksX,hNoOfBlocksY,hNumberOfBlockPerWindowX,hNumberOfBlockPerWindowY;
  int hNumberOfWindowsX,hNumberOfWindowsY;
  double scaleRatio,startScale,endScale,scaleCount;


  int iClosestPowerOfTwo(int x) { x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x++; return x; }

  void header()
  {
  	  printf("\t\t\"ecos\": {\n");
      printf("\t\t\t\"128\": 15.0,\n"); 
      printf("\t\t\t\"256\": 15.0,\n"); 
      printf("\t\t\t\"512\": 15.0,\n"); 
      printf("\t\t\t\"1024\": 15.0,\n"); 
      printf("\t\t\t\"2048\": 15.0,\n"); 
      printf("\t\t\t\"4096\": 15.0,\n"); 
      printf("\t\t\t\"8192\": 15.0\n");
      printf("\t\t},\n");


    printf("\t\t\"id\" :%d\n",id);

    if(id>1) printf("\t\t\"depends\" : %d\n",id-1);

    id++;

    printf("\t\t\"partition\" : %d,\n",partit);

    printf("\t\t\"workdimension\" : \"%d\",\n",workdimension);


    printf("\t\t\"globalWorkSize\": \"[");
    for(int i=0;i<gblworksize;i++) { printf("dataset"); if(i!=gblworksize-1) printf(","); }
    printf("]\"\n");

  	


  	printf("\t\t\"name\" : \"%s\",\n",name);

  	printf("\t\t\"src\" : \"%s\",\n",src);

  	
  	printf("\t\t\"inputfile\" : \"%s\",\n",ip);



  	

  }


  void inputbuffers(int size,int breaks[],int pos[],int dataset[],int typ[],int count[],int flag)
  {

  	/*
  	"inputBuffers": [
          
  	{
              "break": 1, 
              "pos": 3, 
              "size": "dataset**3", 
             "type": "float"
          }    
      ],
      */ 

      if(flag==1) printf("\t\t\"inputBuffers\" : [\n");
      else if(flag==2) printf("\t\t\"outputBuffers\" : [\n");
      else printf("\t\t\"ioBuffers\" : [\n");

      for(int i=0;i<size;i++)
      {
      	printf("\t\t\t{\n");
     		printf("\t\t\t\t\"break\":%d,\n",breaks[i]);
  		  printf("\t\t\t\t\"pos\":%d,\n",pos[i]);
  		  printf("\t\t\t\t\"size\":\"dataset**%d\",\n",dataset[i]);
     		if(count[i]!=0) printf("\t\t\t\t\"type\":\"%s%d\"\n",type[typ[i]],count[i]);
     		else printf("\t\t\t\t\"type\":\"%s\"\n",type[typ[i]]); 

     		if(i!=size-1) printf("\t\t\t},\n");
     		else printf("\t\t\t}\n");


  	}

  	printf("\t\t],\n");


  }

  void varargs(int size,int pos[],int typ[],int count[],float value[][20])
  {

  	printf("\t\t\"varArguments\":[\n");

  	for(int i=0;i<size;i++)
  	{
  		printf("\t\t\t{\n");
  		printf("\t\t\t\t\"pos\":%d,\n",pos[i]);
  		if(count[i]!=0) printf("\t\t\t\t\"type\":%s%d,\n",type[typ[i]],count[i]);
      else printf("\t\t\t\t\"type\":%s,\n",type[typ[i]]);

      if(count[i]!=0)
      {
          printf("\t\t\t\t\"value\":\"[");
          for(int j=0;j<count[i];j++) 
            {
              printf("%.0f",value[i][j]);
              if(j<count[i]-1) printf(",");
            }
            printf("]\"\n");
      
      }
      else
      {
          printf("\t\t\t\t\"value\":\"%.0f\"\n",value[i][0]);
      }

      printf("\t\t\t}");
      if(i<size-1) printf(",\n");
      else printf("\n");
      
  	}

    printf("\t\t]\n\n");


  }

  void initHog(int hWidthROI,int hHeightROI)
  {
      marginX =4 , marginY = 4;
      toaddxx = 0, toaddxy = 0, toaddyx = 0, toaddyy = 0,avSizeX=48,avSizeY=96;
      if (avSizeX) { toaddxx = hWidthROI * marginX / avSizeX; toaddxy = hHeightROI * marginY / avSizeX; }
      if (avSizeY) { toaddyx = hWidthROI * marginX / avSizeY; toaddyy = hHeightROI * marginY / avSizeY; }

      hPaddingSizeX = max(toaddxx, toaddyx), hPaddingSizeY = max(toaddxy, toaddyy);

      hPaddedWidth = hWidthROI + hPaddingSizeX*2;
      hPaddedHeight = hHeightROI+ hPaddingSizeY*2;

      hUseGrayscale = false;

      hNoHistogramBins = 9;
      //hNoHistogramBins = noOfHistogramBins;
      hcellSizeX = 8;  hcellSizeY  =8; hblockSizeX = 2; hblockSizeY = 2;
      //hCellSizeX = cellSizeX; hCellSizeY = cellSizeY; hBlockSizeX = blockSizeX; hBlockSizeY = blockSizeY;
      hwindowSizeX = 64; hwindowSizeY = 128; 
      //hWindowSizeX = windowSizeX; hWindowSizeY = windowSizeY;

      //hNoOfCellsX = 1920/8 = 240 hNoOFCellsY = 1440 /8 =130
      hNoOfCellsX = hPaddedWidth / hcellSizeX;
      hNoOfCellsY = hPaddedHeight / hcellSizeY;

      //hNOOFBlocksX = 240 - 2+1 =239  hBlocksY = 70 - 2 +1 = 69 
      hNoOfBlocksX = hNoOfCellsX - hblockSizeX + 1;
      hNoOfBlocksY = hNoOfCellsY - hblockSizeY + 1;

      //hNumberoFBlockPerWIndowX = 64 -8*2   / 8 +1 =7  
      hNumberOfBlockPerWindowX = (hwindowSizeX - hcellSizeX * hblockSizeX) / hcellSizeX + 1;
      hNumberOfBlockPerWindowY = (hwindowSizeY - hcellSizeY * hblockSizeY) / hcellSizeY + 1;

      hNumberOfWindowsX = 0;
      int i;
      for (i=0; i<hNumberOfBlockPerWindowX; i++) hNumberOfWindowsX += (hNoOfBlocksX-i)/hNumberOfBlockPerWindowX;

      hNumberOfWindowsY = 0;
      for (i=0; i<hNumberOfBlockPerWindowY; i++) hNumberOfWindowsY += (hNoOfBlocksY-i)/hNumberOfBlockPerWindowY;

      scaleRatio = 1.05f;
      startScale = 1.0f;
      endScale = min(hPaddedWidth / (float) hwindowSizeX, hPaddedHeight / (float) hwindowSizeY);
      scaleCount = (int)floor(logf(endScale/startScale)/logf(scaleRatio)) + 1;


  }



  void PadHostImage(int hWidthROI,int hHeightROI)
  {
      

      //we will send Uchar4ToFloat4.cl
      gblworksize = 2;
      partit = 10;
      strcpy(name,"uchar4tofloat4");
      strcpy(src,"uchar4tofloat4.cl");
      workdimension = 2;
      strcpy(ip,"uchar4tofloat4.txt");
      
      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int size=1;
      int breaks[2]={1,1};
      int pos[2]={0,1};
      int dataset[2]={2,0};
      int typ[2]={0,0};
      int count[2]={4,0};
      float value[2][20];

      printf("\t{\n");
      header();
      inputbuffers(size,breaks,pos,dataset,typ,count,1);
      
      breaks[0]=1; 
      pos[0]=1;
      dataset[0]=2;
      typ[0]=5;
      count[0]=4;

      inputbuffers(size,breaks,pos,dataset,typ,count,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      size = 2;
      pos[0]=2; pos[1]=3;
      typ[0]=3; typ[1]=3;
      count[0]=0; count[1]=0;
      value[0][0]=hWidthROI; value[1][0]=hHeightROI; 
      varargs(2,pos,typ,count,value);

      printf("\t}");


  }

  


  void DownScale(int hWidthROI,int hHeightROI,float scale)
  {
      //we will send Uchar4ToFloat4.cl
    if(hUseGrayscale==false)
    {
      gblworksize = 2;
      partit = 10;
      strcpy(name,"resizeFastBicubic1");
      strcpy(src,"resizeFastBicubic1.cl");
      workdimension = 2;
      strcpy(ip,"resizeFastBicubic1.txt");
      

      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int size=1;
      int breaks[3]={1,1};
      int pos[3]={1,1};
      int dataset[3]={2,0};
      int typ[3]={5,0};
      int count[3]={4,0};
      float value[3][20];

      printf("\t{\n");
      header();
      inputbuffers(size,breaks,pos,dataset,typ,count,1);
      
      breaks[0]=1; 
      pos[0]=0;
      dataset[0]=2;
      typ[0]=5;
      count[0]=4;

      inputbuffers(size,breaks,pos,dataset,typ,count,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      size = 3;
      pos[0]=2; pos[1]=3; pos[2]=4;
      typ[0]=3; typ[1]=3; typ[2]=5;
      count[0]=0; count[1]=0; count[2]=0;
      value[0][0]=hWidthROI; value[1][0]=hHeightROI;  value[2][0]=scale;
      varargs(3,pos,typ,count,value);

      printf("\t}");

    }
    else
    {
      gblworksize = 2;
      partit = 10;
      strcpy(name,"resizeFastBicubic1");
      strcpy(src,"resizeFastBicubic1.cl");
      workdimension = 2;
      strcpy(ip,"resizeFastBicubic1.txt");
      

      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int size=1;
      int breaks[3]={1,1};
      int pos[3]={1,1};
      int dataset[3]={2,0};
      int typ[3]={5,0};
      int count[3]={1,0};
      float value[3][20];

      printf("\t{\n");
      header();
      inputbuffers(size,breaks,pos,dataset,typ,count,1);
      
      breaks[0]=1; 
      pos[0]=0;
      dataset[0]=2;
      typ[0]=5;
      count[0]=1;

      inputbuffers(size,breaks,pos,dataset,typ,count,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      size = 3;
      pos[0]=2; pos[1]=3; pos[2]=4;
      typ[0]=3; typ[1]=3; typ[2]=5;
      count[0]=0; count[1]=0; count[2]=0;
      value[0][0]=hWidthROI; value[1][0]=hHeightROI;  value[2][0]=scale;
      varargs(3,pos,typ,count,value);

      printf("\t}");
    }


  }



  void ConvolutionImage(int hWidthROI,int hHeightROI)
  {
      //we will send Uchar4ToFloat4.cl
    if(hUseGrayscale==false)
    {

      //convolutionRpwGPU4.cl
      gblworksize = 2;
      partit = 10;
      strcpy(name,"convolutionRowGPU4");
      strcpy(src,"convolutionRowGPU4.cl");
      workdimension = 2;
      strcpy(ip,"convolutionRowGPU4.txt");
      

      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int size=1;
      int breaks[3]={1,1};
      int pos[3]={1,1};
      int dataset[3]={2,0};
      int typ[3]={5,0};
      int count[3]={4,0};
      float value[3][20];

      printf("\t{\n");
      header();
      inputbuffers(size,breaks,pos,dataset,typ,count,1);
      

      breaks[0]=1; 
      pos[0]=0;
      dataset[0]=2;
      typ[0]=5;
      count[0]=4;

      inputbuffers(size,breaks,pos,dataset,typ,count,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      size = 2;
      pos[0]=2; pos[1]=3; 
      typ[0]=3; typ[1]=3; 
      count[0]=0; count[1]=0; 
      value[0][0]=hWidthROI; value[1][0]=hHeightROI; 
      varargs(2,pos,typ,count,value);

      printf("\t},\n\n");

 //convolutionColumnGPU4 to 2.opencl
      gblworksize = 2;
      partit = 10;
      strcpy(name,"convolutionColumnGPU4to2 ");
      strcpy(src,"convolutionColumnGPU4to2.cl");
      workdimension = 2;
      strcpy(ip,"convolutionColumnGPU4to2.txt");
      

      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int sizes=2;
      int breakss[4]={1,1,1};
      int poss[4]={1,2,1};
      int datasets[4]={2,2,0};
      int typs[4]={5,5,0};
      int counts[4]={4,4,0};
      float values[4][20];

      printf("\t{\n");
      header();
      inputbuffers(sizes,breakss,poss,datasets,typs,counts,1);
      
      sizes=1;
      breakss[0]=1; 
      poss[0]=0;
      datasets[0]=2;
      typs[0]=5;
      counts[0]=2;

      inputbuffers(sizes,breakss,poss,datasets,typs,counts,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      sizes= 4;
      poss[0]=3; poss[1]=4; poss[2]=5; poss[3]=6;
      typs[0]=3; typs[1]=3; typs[2]=3; typs[3]=3;
      counts[0]=0; counts[1]=0; counts[2]=0; counts[3]=0;
      values[0][0]=hWidthROI; values[1][0]=hHeightROI; values[2][0]=256 ; values[3][0]=hWidthROI*8;
      varargs(2,poss,typs,counts,values);

      printf("\t}");






    }
    else
    {
       //convolutionRpwGPU1.cl
      gblworksize = 2;
      partit = 10;
      strcpy(name,"convolutionRowGPU1");
      strcpy(src,"convolutionRowGPU1.cl");
      workdimension = 2;
      strcpy(ip,"convolutionRowGPU1.txt");
      

      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int size=1;
      int breaks[3]={1,1};
      int pos[3]={1,1};
      int dataset[3]={2,0};
      int typ[3]={5,0};
      int count[3]={1,0};
      float value[3][20];

      printf("\t{\n");
      header();
      inputbuffers(size,breaks,pos,dataset,typ,count,1);
      
      breaks[0]=1; 
      pos[0]=0;
      dataset[0]=2;
      typ[0]=5;
      count[0]=1;

      inputbuffers(size,breaks,pos,dataset,typ,count,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      size = 2;
      pos[0]=2; pos[1]=3; 
      typ[0]=3; typ[1]=3; 
      count[0]=0; count[1]=0; 
      value[0][0]=hWidthROI; value[1][0]=hHeightROI; 
      varargs(2,pos,typ,count,value);

      printf("\t},\n\n");

      //convolutionColumnGPU4 to 2.opencl
      gblworksize = 2;
      partit = 10;
      strcpy(name,"convolutionColumnGPU1to2 ");
      strcpy(src,"convolutionColumnGPU1to2.cl");
      workdimension = 2;
      strcpy(ip,"convolutionColumnGPU1to2.txt");
      

      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int sizes=2;
      int breakss[4]={1,1,1};
      int poss[4]={1,2,1};
      int datasets[4]={2,2,0};
      int typs[4]={5,5,0};
      int counts[4]={1,1,0};
      float values[4][20];

      printf("\t{\n");
      header();
      inputbuffers(sizes,breakss,poss,datasets,typs,counts,1);
      
      sizes = 1;
      breakss[0]=1; 
      poss[0]=0;
      datasets[0]=2;
      typs[0]=5;
      counts[0]=2;

      inputbuffers(sizes,breakss,poss,datasets,typs,counts,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      sizes= 4;
      poss[0]=3; poss[1]=4; poss[2]=5; poss[3]=6;
      typs[0]=3; typs[1]=3; typs[2]=3; typs[3]=3;
      counts[0]=0; counts[1]=0; counts[2]=0; counts[3]=0;
      values[0][0]=hWidthROI; values[1][0]=hHeightROI; values[2][0]=256 ; values[3][0]=hWidthROI*8;
      varargs(2,poss,typs,counts,values);

      printf("\t}");


    }


  }

  void computeblockHistograms(int width, int height)
  {
  
    int rNoOfCellsX = width / hcellSizeX;  //1920 /8 = 240
    int rNoOfCellsY = height / hcellSizeY;  //1440/8 =160

    int rNoOfBlocksX = rNoOfCellsX - hblockSizeX + 1; //239
    int rNoOfBlocksY = rNoOfCellsY - hblockSizeY + 1; //159

    int rNumberOfWindowsX = (width-hwindowSizeX)/hcellSizeX + 1;  // (240-64)/8 +1 = 23
    int rNumberOfWindowsY = (height-hwindowSizeY)/hcellSizeY + 1; //160 - 128) /8 +1  = 5

    int leftoverX = (width - hwindowSizeX - hcellSizeX * (rNumberOfWindowsX - 1))/2;  //=  ( 1920 -64 - 8*22 )/2 = 840
    int leftoverY = (height - hwindowSizeY - hcellSizeY * (rNumberOfWindowsY - 1))/2; // = 1440 -128 - 8*4 ) /2 =640


    
  //computeBlockHistogramsWithGauss
    //(inputImage, blockHistograms, noHistogramBins, cellSizeX, cellSizeY, blockSizeX, blockSizeY, leftoverX, leftoverY, width, height);
    //image , float array , 9 , 8 , 8 , 2  , 2 ,64 ,128 , 840 , 640 ,1920 ,1440 

  
      gblworksize = 2;
      partit = 10;
      strcpy(name,"computeBlockHistogramsWithGauss");
      strcpy(src,"computeBlockHistogramsWithGauss.cl");
      workdimension = 2;
      strcpy(ip,"computeBlockHistogramsWithGauss.txt");
      
      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int size=1;
      int breaks[10]={1,1};
      int pos[10]={0,1};
      int dataset[10]={3,0};
      int typ[10]={5,0};
      int count[10]={2,0};
      float value[10][20];

      printf("\t{\n");
      header();
      inputbuffers(size,breaks,pos,dataset,typ,count,1);
      

      breaks[0]=1; 
      pos[0]=1;
      dataset[0]=2;
      typ[0]=5;
      count[0]=1;

      inputbuffers(size,breaks,pos,dataset,typ,count,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      size = 9;
      pos[0]=2; pos[1]=3; pos[2]=4 ;pos[3]=5 ;pos[4]=6 ;pos[5]=7 ;pos[6]=8 ;pos[7]=9 ;pos[8]=10 ;
      typ[0]=3; typ[1]=3; typ[2]=3 ;typ[3]= 3;typ[4]=3 ;typ[5]=3 ;typ[6]=3 ;typ[7]=3 ;typ[8]=3 ;
      count[0]=0; count[1]=0; count[2]=0 ;count[3]=0; count[4]=0; count[5]=0; count[6]=0; count[7]=0; count[8]=0;
      value[0][0]= 9; value[1][0]=8; value[2][0]=8; value[3][0]= 2;value[4][0]=2; value[5][0]=leftoverX ; value[6][0]=leftoverY; value[7][0]=width ; value[8][0] = height;
      //9 , 8 , 8 , 2  , 2 ,64 ,128
      varargs(9,pos,typ,count,value);

      printf("\t}");



  }


  void NormalizeblockHistograms(int width, int height)
  {
  
    int rNoOfCellsX = width / hcellSizeX;
    int rNoOfCellsY = height / hcellSizeY;

    int rNoOfBlocksX = rNoOfCellsX - hblockSizeX + 1;
    int rNoOfBlocksY = rNoOfCellsY - hblockSizeY + 1;

    int alignedBlockDimX = iClosestPowerOfTwo(hNoHistogramBins);
    int alignedBlockDimY = iClosestPowerOfTwo(hblockSizeX);
    int alignedBlockDimZ = iClosestPowerOfTwo(hblockSizeY);

    
    //  __global__ void normalizeBlockHistograms(float1 *blockHistograms, int noHistogramBins,
//                     int rNoOfHOGBlocksX, int rNoOfHOGBlocksY,
//                     int blockSizeX, int blockSizeY,
//                     int alignedBlockDimX, int alignedBlockDimY, int alignedBlockDimZ,
//                     int width, int height)
  
      gblworksize = 2;
      partit = 10;
      strcpy(name,"normalizeBlockHistograms");
      strcpy(src,"normalizeBlockHistograms.cl");
      workdimension = 2;
      strcpy(ip,"normalizeBlockHistograms.txt");
      
      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int size=1;
      int breaks[11]={1,1};
      int pos[11]={0,1};
      int dataset[11]={3,0};
      int typ[11]={5,0};
      int count[11]={2,0};
      float value[11][20];

      printf("\t{\n");
      header();
      inputbuffers(size,breaks,pos,dataset,typ,count,3);
      
      // breaks[0]=1; 
      // pos[0]=1;
      // dataset[0]=2;
      // typ[0]=5;
      // count[0]=1;

      // inputbuffers(size,breaks,pos,dataset,typ,count,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      size = 10;
      pos[0]=1; pos[1]=2; pos[2]=3 ;pos[3]=4 ;pos[4]=5 ;pos[5]=6 ;pos[6]=7 ;pos[7]=8 ;pos[8]=9 ; pos[9]=10;
      typ[0]=3; typ[1]=3; typ[2]=3 ;typ[3]= 3;typ[4]=3 ;typ[5]=3 ;typ[6]=3 ;typ[7]=3 ;typ[8]=3 ; typ[9]=3;
      count[0]=0; count[1]=0; count[2]=0 ;count[3]=0; count[4]=0; count[5]=0; count[6]=0; count[7]=0; count[8]=0; count[9]=0;
      value[0][0]= 9; value[1][0]= rNoOfBlocksX; value[2][0]=rNoOfBlocksY; value[3][0]= hblockSizeX; value[4][0]=hblockSizeY; 
      value[5][0]=alignedBlockDimX ; value[6][0]=alignedBlockDimY; value[7][0]=alignedBlockDimZ; value[8][0]=width ; value[9][0] = height;
      //9 , 8 , 8 , 2  , 2 ,64 ,128
      varargs(10,pos,typ,count,value);

      printf("\t}");



  }


  void computeSvmScores(int width, int height,int iter)
  {


    

      gblworksize = 2;
      partit = 10;
      strcpy(name," LinearSVMEvaluation");
      strcpy(src," LinearSVMEvaluation.cl");
      workdimension = 2;
      strcpy(ip," LinearSVMEvaluation.txt");
      
      ///void inputbuffers(int size,int breaks[],int pos[],int dataset[],int count[],int flag)

      int size=2;
      int breaks[20]={1,1};
      int pos[20]={1,2,1};
      int dataset[20]={2,2,0};
      int typ[20]={5,5,0};
      int count[20]={0,1,0};
      float value[20][20];

      printf("\t{\n");
      header();
      inputbuffers(size,breaks,pos,dataset,typ,count,1);
      
      size=1;
      breaks[0]=1; 
      pos[0]=0;
      dataset[0]=2;
      typ[0]=5;
      count[0]=1;

      inputbuffers(size,breaks,pos,dataset,typ,count,0);

      //void varargs(int size,int pos[],int typ[],int count[],int value[][20])
      size = 18;

      for(int i=0;i<18;i++) 
      {
        pos[i]=i+3;
        typ[i]=3;
        count[i]=0;
      }

      /*    
       
            // 9 ,  64 ,128 ,8,  8 , 2 , 2  ,  ,  , i , width , height
              LinearSVMEvaluation(svmScores, blockHistograms, hNoHistogramBins, hWindowSizeX, hWindowSizeY, hCellSizeX, hCellSizeY,
              hBlockSizeX, hBlockSizeY, rNoOfBlocksX, rNoOfBlocksY, i, rPaddedWidth, rPaddedHeight);



            __host__ void LinearSVMEvaluation(float1* svmScores, float1* blockHistograms, int noHistogramBins,
                  int windowSizeX, int windowSizeY,
                  int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
                  int hogBlockCountX, int hogBlockCountY,
                  int scaleId, int width, int height)


            linearSVMEvaluation
                  (svmScores, svmBias, blockHistograms, noHistogramBins,
                   windowSizeX, windowSizeY, hogBlockCountX, hogBlockCountY, cellSizeX, cellSizeY,
                   hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY,
                   blockSizeX, blockSizeY, alignedBlockDimX, scaleId, scaleCount,
                   hNumberOfWindowsX, hNumberOfWindowsY, width, height);


            
        __kernel void linearSVMEvaluation(__global float1* svmScores,
          __global float svmBias,
          __global float1* blockHistograms,
          const int noHistogramBins,
          const int windowSizeX,
          const int windowSizeY,
          const int hogBlockCountX,
          const int hogBlockCountY,
          const int cellSizeX,
          const int cellSizeY,
          const int numberOfBlockPerWindowX,
          const int numberOfBlockPerWindowY,
          const int blockSizeX,
          const int blockSizeY,
          const int alignedBlockDimX,
          const int scaleId,
          const int scaleCount,
          const int hNumberOfWindowsX,
          const int hNumberOfWindowsY,
          const int width,
          const int height)

      
      */

      int rNoOfCellsX = width / hcellSizeX;  //1920 /8 = 240
      int rNoOfCellsY = height / hcellSizeY;  //1440/8 =160

      int rNoOfBlocksX = rNoOfCellsX - hblockSizeX + 1; //239
      int rNoOfBlocksY = rNoOfCellsY - hblockSizeY + 1; //159
  
      int rNumberOfWindowsX = (width-hwindowSizeX)/hcellSizeX + 1;
      int rNumberOfWindowsY = (height-hwindowSizeY)/hcellSizeY + 1;

  
      int alignedBlockDimX = iClosestPowerOfTwo(hNoHistogramBins * hblockSizeX * hNumberOfBlockPerWindowX);

      value[0][0]= 9; value[1][0]= hwindowSizeX; value[2][0]=hwindowSizeY; value[3][0]= rNoOfBlocksX; value[4][0]=rNoOfBlocksY; 
      value[6][0]=hcellSizeX; value[7][0]=hcellSizeY; value[8][0]=hNumberOfBlockPerWindowX ; value[9][0] = hNumberOfBlockPerWindowY;
      value[10][0]=hblockSizeX ; value[11][0]=  hblockSizeY; value[12][0]= alignedBlockDimX; value[13][0]=iter; value[14][0] =scaleCount; rNumberOfWindowsX;
      value[15][0]=rNumberOfWindowsY; value[16][0]=width; value[17][0]=height; 
      

      //9 , 8 , 8 , 2  , 2 ,64 ,128
      varargs(18,pos,typ,count,value);

      printf("\t}");

  }

  void def(int width, int height)
  {
    initHog(width,height);
    printf("{\n");
  	PadHostImage(hPaddedWidth,hPaddedHeight);
    printf(",\n\n");

    float scale =1.05f;
    for(int i=0;i<scaleCount;i++)
    {
        DownScale(hPaddedWidth,hPaddedHeight,scale);
        
        printf(",");
        printf("\n\n");

        ConvolutionImage(hPaddedWidth,hPaddedHeight);

        printf(",");
        printf("\n\n");


        computeblockHistograms(hPaddedWidth,hPaddedHeight);


        printf(",");
        printf("\n\n");

        NormalizeblockHistograms(hPaddedWidth,hPaddedHeight);

        printf(",");
        printf("\n\n");

        computeSvmScores(hPaddedWidth,hPaddedHeight,i);

        if(i<scaleCount-1)  printf(",");
        printf("\n\n");

        scale = scale*(1.05f);
        



    }

    printf("}\n");


  }

  int main()
  {
      HOGImage* image = new HOGImage("testImage.bmp");
      HOGImage* imageCUDA = new HOGImage(image->width,image->height);

      def(image->width,image->height);

  }


  