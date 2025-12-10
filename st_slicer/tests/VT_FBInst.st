PROGRAM VT_FBInst

VAR
CV_Canny1 : CV_Canny;
CV_Canny1_OutWidth : UINT;
CV_Canny1_OutHeight : UINT;
CV_Canny1_OutChannel : UINT;
CV_Canny1_Execute : BOOL;
CV_Canny1_Done : BOOL;
IMG_Resize1 : IMG_Resize;
IMG_Resize1_ResizeW : UINT;
IMG_Resize1_ResizeH : UINT;
IMG_Resize1_Execute : BOOL;
IMG_Resize1_Done : BOOL;
IMG_Resize2 : IMG_Resize;
IMG_Resize2_ResizeW : UINT;
IMG_Resize2_ResizeH : UINT;
IMG_Resize2_Execute : BOOL;
IMG_Resize2_Done : BOOL;
APP_AdjustPlaneZ1 : APP_AdjustPlaneZ;
APP_AdjustPlaneZ1_Execute : BOOL;
APP_AdjustPlaneZ1_Done : BOOL;
APP_AdjustPlaneZ1_EN : BOOL;
APP_CalBestPlank_ST1 : APP_CalBestPlank_ST;
APP_CalBestPlank_ST1_Execute : BOOL;
APP_CalBestPlank_ST1_Done : BOOL;
APP_CalBestPlank_ST1_EN : BOOL;
APP_ClassfyPolysByTags1 : APP_ClassfyPolysByTags;
APP_ClassfyPolysByTags1_Execute : BOOL;
APP_ClassfyPolysByTags1_Done : BOOL;
APP_ClassfyPolysByTags1_EN : BOOL;
APP_DrawContours1 : APP_DrawContours;
APP_DrawContours1_Execute : BOOL;
APP_DrawContours1_Done : BOOL;
APP_DrawContours1_EN : BOOL;
APP_DrawDivisionRects1 : APP_DrawDivisionRects;
APP_DrawDivisionRects1_Execute : BOOL;
APP_DrawDivisionRects1_Done : BOOL;
APP_DrawDivisionRects1_EN : BOOL;
APP_FilterColorHSV1 : APP_FilterColorHSV;
APP_FilterColorHSV1_Execute : BOOL;
APP_FilterColorHSV1_Done : BOOL;
APP_FilterColorHSV1_EN : BOOL;
APP_FilterColorHSV2 : APP_FilterColorHSV;
APP_FilterColorHSV2_Execute : BOOL;
APP_FilterColorHSV2_Done : BOOL;
APP_FilterColorHSV2_EN : BOOL;
APP_FilterColorHSV3 : APP_FilterColorHSV;
APP_FilterColorHSV3_Execute : BOOL;
APP_FilterColorHSV3_Done : BOOL;
APP_FilterColorHSV3_EN : BOOL;
APP_FilterPlankBySize1 : APP_FilterPlankBySize;
APP_FilterPlankBySize1_Execute : BOOL;
APP_FilterPlankBySize1_Done : BOOL;
APP_FilterPlankBySize1_EN : BOOL;
APP_FindConRects1 : APP_FindConRects;
APP_FindConRects1_Execute : BOOL;
APP_FindConRects1_Done : BOOL;
APP_FindConRects1_EN : BOOL;
APP_FindConRects2 : APP_FindConRects;
APP_FindConRects2_Execute : BOOL;
APP_FindConRects2_Done : BOOL;
APP_FindConRects2_EN : BOOL;
APP_FindContoursVertex1 : APP_FindContoursVertex;
APP_FindContoursVertex1_Execute : BOOL;
APP_FindContoursVertex1_Done : BOOL;
APP_FindContoursVertex1_EN : BOOL;
APP_GetMinAreaRect1 : APP_GetMinAreaRect;
APP_GetMinAreaRect1_Execute : BOOL;
APP_GetMinAreaRect1_Done : BOOL;
APP_GetMinAreaRect1_EN : BOOL;
APP_LabelsIntersection1 : APP_LabelsIntersection;
APP_LabelsIntersection1_Execute : BOOL;
APP_LabelsIntersection1_Done : BOOL;
APP_LabelsIntersection1_EN : BOOL;
APP_LocatePullCorner1 : APP_LocatePullCorner;
APP_LocatePullCorner1_Execute : BOOL;
APP_LocatePullCorner1_CornerEdgeMinLen : REAL;
APP_LocatePullCorner1_PullDist : REAL;
APP_LocatePullCorner1_Done : BOOL;
APP_LocatePullCorner1_PlanNum : UINT;
APP_LocatePullCorner1_EN : BOOL;
APP_LocateSuckCorner1 : APP_LocateSuckCorner2;
APP_LocateSuckCorner1_Execute : BOOL;
APP_LocateSuckCorner1_Done : BOOL;
APP_LocateSuckCorner1_EN : BOOL;
APP_MaskImage1 : APP_MaskImage;
APP_MaskImage1_Execute : BOOL;
APP_MaskImage1_Done : BOOL;
APP_MaskImage1_EN : BOOL;
APP_MaskImage2 : APP_MaskImage;
APP_MaskImage2_Execute : BOOL;
APP_MaskImage2_Done : BOOL;
APP_MaskImage2_EN : BOOL;
APP_MoodDivision1 : APP_MoodDivision;
APP_MoodDivision1_Execute : BOOL;
APP_MoodDivision1_Done : BOOL;
APP_MoodDivision1_EN : BOOL;
APP_PointOffset2 : APP_PointsOffset;
APP_PointOffset2_Execute : BOOL;
APP_PointOffset2_Done : BOOL;
APP_PointOffset2_EN : BOOL;
APP_PointsOffset1 : APP_PointsOffset;
APP_PointsOffset1_Execute : BOOL;
APP_PointsOffset1_Done : BOOL;
APP_PointsOffset1_EN : BOOL;
APP_PointsRotate1 : APP_PointsRotate;
APP_PointsRotate1_Execute : BOOL;
APP_PointsRotate1_Done : BOOL;
APP_PointsRotate1_EN : BOOL;
APP_PointsRotate2 : APP_PointsRotate;
APP_PointsRotate2_Execute : BOOL;
APP_PointsRotate2_Done : BOOL;
APP_PointsRotate2_EN : BOOL;
APP_ReadImage1 : APP_ReadImage;
APP_ReadImage1_Execute : BOOL;
APP_ReadImage1_Done : BOOL;
APP_ReadImage1_EN : BOOL;
APP_ReadImage2 : APP_ReadImage;
APP_ReadImage2_Execute : BOOL;
APP_ReadImage2_Done : BOOL;
APP_ReadImage2_EN : BOOL;
APP_RectsAverOverlay1 : APP_RectsAverOverlay;
APP_RectsAverOverlay1_Execute : BOOL;
APP_RectsAverOverlay1_OverlayThres : REAL;
APP_RectsAverOverlay1_Done : BOOL;
APP_RectsAverOverlay1_EN : BOOL;
APP_RectsOverlay1 : APP_RectsOverlay;
APP_RectsOverlay1_Execute : BOOL;
APP_RectsOverlay1_OverlayThres : REAL;
APP_RectsOverlay1_MoreOrLess : BOOL;
APP_RectsOverlay1_Done : BOOL;
APP_RectsOverlay1_EN : BOOL;
APP_RectsOverlay3 : APP_RectsOverlay;
APP_RectsOverlay3_Execute : BOOL;
APP_RectsOverlay3_OverlayThres : REAL;
APP_RectsOverlay3_MoreOrLess : BOOL;
APP_RectsOverlay3_Done : BOOL;
APP_RectsOverlay3_EN : BOOL;
APP_SaveImage1 : APP_SaveImage;
APP_SaveImage1_Execute : BOOL;
APP_SaveImage1_Done : BOOL;
APP_SaveImage1_EN : BOOL;
APP_SaveImage2 : APP_SaveImage;
APP_SaveImage2_Execute : BOOL;
APP_SaveImage2_Done : BOOL;
APP_SaveImage2_EN : BOOL;
APP_SaveImage3 : APP_SaveImage;
APP_SaveImage3_Execute : BOOL;
APP_SaveImage3_Done : BOOL;
APP_SaveImage3_Error : BOOL;
APP_SaveImage3_EN : BOOL;
APP_SaveImage4 : APP_SaveImage;
APP_SaveImage4_Execute : BOOL;
APP_SaveImage4_InChannel : UINT;
APP_SaveImage4_Done : BOOL;
APP_SaveImage4_Error : BOOL;
APP_SaveImage4_EN : BOOL;
APP_SaveImage5 : APP_SaveImage;
APP_SaveImage5_Execute : BOOL;
APP_SaveImage5_Done : BOOL;
APP_SaveImage5_Error : BOOL;
APP_SaveImage5_EN : BOOL;
APP_SaveImage6 : APP_SaveImage;
APP_SaveImage6_Execute : BOOL;
APP_SaveImage6_Done : BOOL;
APP_SaveImage6_Error : BOOL;
APP_SaveImage6_EN : BOOL;
APP_SetRightContours1 : APP_SetRightContours;
APP_SetRightContours1_Execute : BOOL;
APP_SetRightContours1_Done : BOOL;
APP_SetRightContours1_EN : BOOL;
APP_ThresholdImg1 : APP_ThresholdImg;
APP_ThresholdImg1_Execute : BOOL;
APP_ThresholdImg1_Done : BOOL;
APP_ThresholdImg1_EN : BOOL;
APP_TransToolAbsPos2D1 : APP_TransToolAbsPos2D;
APP_TransToolAbsPos2D1_Execute : BOOL;
APP_TransToolAbsPos2D1_Done : BOOL;
APP_TransToolAbsPos2D1_EN : BOOL;
APP_TransToolAbsPos2D2 : APP_TransToolAbsPos2D;
APP_TransToolAbsPos2D2_Execute : BOOL;
APP_TransToolAbsPos2D2_Done : BOOL;
APP_TransToolAbsPos2D2_EN : BOOL;
APP_UINT2REAL1 : APP_UINT2REAL;
APP_UINT2REAL1_Execute : BOOL;
APP_UINT2REAL1_Done : BOOL;
APP_UINT2REAL1_EN : BOOL;
CV_PolyMask1 : CV_PolyMask;
CV_PolyMask1_Flag : DINT;
CV_PolyMask1_Fill : BOOL;
CV_PolyMask1_OutWidth : UINT;
CV_PolyMask1_OutHeight : UINT;
CV_PolyMask1_OutChannel : UINT;
CV_PolyMask1_Execute : BOOL;
CV_PolyMask1_Done : BOOL;
CV_PolyMask2 : CV_PolyMask;
CV_PolyMask2_Flag : DINT;
CV_PolyMask2_Fill : BOOL;
CV_PolyMask2_OutWidth : UINT;
CV_PolyMask2_OutHeight : UINT;
CV_PolyMask2_OutChannel : UINT;
CV_PolyMask2_Execute : BOOL;
CV_PolyMask2_Done : BOOL;
GRPH_RectInfo1 : GRPH_GetRectInfo;
GRPH_RectInfo1_Execute : BOOL;
GRPH_RectInfo1_Done : BOOL;
GRPH_RectInfo1_EN : BOOL;
Mood_ContoursExtract1 : Mood_ContoursExtract;
Mood_ContoursExtract1_Execute : BOOL;
Mood_ContoursExtract1_Done : BOOL;
Mood_ContoursExtract1_EN : BOOL;
Mood_EdgeExtract1 : Mood_EdgeExtract;
Mood_EdgeExtract1_Execute : BOOL;
Mood_EdgeExtract1_Done : BOOL;
Mood_EdgeExtract1_EN : BOOL;
Mood_EdgeExtract2 : Mood_EdgeExtract;
Mood_EdgeExtract2_Execute : BOOL;
Mood_EdgeExtract2_Done : BOOL;
Mood_EdgeExtract2_EN : BOOL;
Mood_PerspTransImage1 : Mood_PerspTransImage;
Mood_PerspTransImage1_Execute : BOOL;
Mood_PerspTransImage1_Done : BOOL;
Mood_PerspTransImage1_EN : BOOL;
Mood_PerspTransImage2 : Mood_PerspTransImage;
Mood_PerspTransImage2_Execute : BOOL;
Mood_PerspTransImage2_Done : BOOL;
Mood_PerspTransImage2_EN : BOOL;
Mood_PerspTransImage3 : Mood_PerspTransImage;
Mood_PerspTransImage3_Execute : BOOL;
Mood_PerspTransImage3_Done : BOOL;
Mood_PerspTransImage3_EN : BOOL;
Mood_PerspTransImage4 : Mood_PerspTransImage;
Mood_PerspTransImage4_Execute : BOOL;
Mood_PerspTransImage4_Done : BOOL;
Mood_PerspTransImage4_EN : BOOL;
Mood_PerspTransImage5 : Mood_PerspTransImage;
Mood_PerspTransImage5_Execute : BOOL;
Mood_PerspTransImage5_Done : BOOL;
Mood_PerspTransImage5_DstWidth : UINT;
Mood_PerspTransImage5_DstHeight : UINT;
Mood_PerspTransImage5_DstChannel : UINT;
Mood_PerspTransImage5_OffsetX : REAL;
Mood_PerspTransImage5_OffsetY : REAL;
Mood_PerspTransImage5_EN : BOOL;
Mood_PerspTransImage6 : Mood_PerspTransImage;
Mood_PerspTransImage6_Execute : BOOL;
Mood_PerspTransImage6_Done : BOOL;
Mood_PerspTransImage6_DstWidth : UINT;
Mood_PerspTransImage6_DstHeight : UINT;
Mood_PerspTransImage6_DstChannel : UINT;
Mood_PerspTransImage6_OffsetX : REAL;
Mood_PerspTransImage6_OffsetY : REAL;
Mood_PerspTransImage6_EN : BOOL;
Mood_PerspTransImage7 : Mood_PerspTransImage;
Mood_PerspTransImage7_Execute : BOOL;
Mood_PerspTransImage7_Done : BOOL;
Mood_PerspTransImage7_DstWidth : UINT;
Mood_PerspTransImage7_DstHeight : UINT;
Mood_PerspTransImage7_DstChannel : UINT;
Mood_PerspTransImage7_OffsetX : REAL;
Mood_PerspTransImage7_OffsetY : REAL;
Mood_PerspTransImage7_EN : BOOL;
Mood_PerspTransImage8 : Mood_PerspTransImage;
Mood_PerspTransImage8_Execute : BOOL;
Mood_PerspTransImage8_Done : BOOL;
Mood_PerspTransImage8_DstWidth : UINT;
Mood_PerspTransImage8_DstHeight : UINT;
Mood_PerspTransImage8_DstChannel : UINT;
Mood_PerspTransImage8_OffsetX : REAL;
Mood_PerspTransImage8_OffsetY : REAL;
Mood_PerspTransImage8_EN : BOOL;
Temp_ScaleParallelogram1 : Temp_ScaleParallelogram;
Temp_ScaleParallelogram1_Execute : BOOL;
Temp_ScaleParallelogram1_WidthParameter : REAL;
Temp_ScaleParallelogram1_HeightParameter : REAL;
Temp_ScaleParallelogram1_IsScale : BOOL;
Temp_ScaleParallelogram1_Done : BOOL;
Temp_ScaleParallelogram1_EN : BOOL;
Temp_ScaleParallelogram2 : Temp_ScaleParallelogram;
Temp_ScaleParallelogram2_Execute : BOOL;
Temp_ScaleParallelogram2_WidthParameter : REAL;
Temp_ScaleParallelogram2_HeightParameter : REAL;
Temp_ScaleParallelogram2_IsScale : BOOL;
Temp_ScaleParallelogram2_Done : BOOL;
Temp_ScaleParallelogram2_EN : BOOL;




APP_FindCenterByRay1 : APP_FindCenterByRay;
APP_FindCenterByRay1_Execute : BOOL;
APP_FindCenterByRay1_TypeNum : DINT;
APP_FindCenterByRay1_ErrorDistance : DINT;
APP_FindCenterByRay1_Epsilon1 : REAL;
APP_FindCenterByRay1_Epsilon2 : REAL;
APP_FindCenterByRay1_ThetaLimit : REAL;
APP_FindCenterByRay1_Done : BOOL;
APP_FindCenterByRay1_OutWidth : UINT;
APP_FindCenterByRay1_OutHeight : UINT;
APP_FindCenterByRay1_OutChannel : UINT;
APP_FindCenterByRay1_EN : BOOL;

APP_ContourClosetSort1 : APP_ContourClosetSort;
APP_ContourClosetSort1_Execute : BOOL;
APP_ContourClosetSort1_Done : BOOL;
APP_ContourClosetSort1_EN : BOOL;

APP_RectsAverOverlay_Tag : APP_RectsAverOverlay;
APP_RectsAverOverlay_Tag_Execute : BOOL;
APP_RectsAverOverlay_Tag_Done : BOOL;
APP_RectsAverOverlay_Tag_EN : BOOL;
Cam1_FindTagContours1 : APP_FindConRects;
Cam1_FindTagContours1_Execute : BOOL;
Cam1_FindTagContours1_SrcChannel : UINT;
Cam1_FindTagContours1_Done : BOOL;
Cam1_FindTagContours1_EN : BOOL;
Cam2_FindTagContours1 : APP_FindConRects;
Cam2_FindTagContours1_Execute : BOOL;
Cam2_FindTagContours1_SrcChannel : UINT;
Cam2_FindTagContours1_Done : BOOL;
Cam2_FindTagContours1_EN : BOOL;
Cam1_CannyEdge : CV_Canny;
Cam1_CannyEdge_InChannel : UINT;
Cam1_CannyEdge_KernelSize : UINT;
Cam1_CannyEdge_OutWidth : UINT;
Cam1_CannyEdge_OutHeight : UINT;
Cam1_CannyEdge_OutChannel : UINT;
Cam1_CannyEdge_Execute : BOOL;
Cam1_CannyEdge_Done : BOOL;
Cam2_CannyEdge : CV_Canny;
Cam2_CannyEdge_KernelSize : UINT;
Cam2_CannyEdge_OutWidth : UINT;
Cam2_CannyEdge_OutHeight : UINT;
Cam2_CannyEdge_OutChannel : UINT;
Cam2_CannyEdge_Execute : BOOL;
Cam2_CannyEdge_Done : BOOL;



CV_EdgeAddDilate : CV_Dilate;
CV_EdgeAddDilate_InChannel : UINT;
CV_EdgeAddDilate_OutWidth : UINT;
CV_EdgeAddDilate_OutHeight : UINT;
CV_EdgeAddDilate_OutChannel : UINT;
CV_EdgeAddDilate_Execute : BOOL;
CV_EdgeAddDilate_Done : BOOL;
CV_FindPlankContours : CV_FindAllContoursAndRect;
CV_FindPlankContours_InChannel : UINT;
CV_FindPlankContours_OutWidth : UINT;
CV_FindPlankContours_OutHeight : UINT;
CV_FindPlankContours_OutChannel : UINT;
CV_FindPlankContours_Execute : BOOL;
CV_FindPlankContours_Done : BOOL;
Cam1And2_EdgeAdd : CV_Or;
Cam1And2_EdgeAdd_InChannel : UINT;
Cam1And2_EdgeAdd_OutWidth : UINT;
Cam1And2_EdgeAdd_OutHeight : UINT;
Cam1And2_EdgeAdd_OutChannel : UINT;
Cam1And2_EdgeAdd_Execute : BOOL;
Cam1And2_EdgeAdd_Done : BOOL;
Cam1_Perspective1 : CV_Perspective2;
Cam1_Perspective1_Execute : BOOL;
Cam1_Perspective1_Done : BOOL;
Cam2_Perspective1 : CV_Perspective2;
Cam2_Perspective1_Execute : BOOL;
Cam2_Perspective1_Done : BOOL;
Cam1PlaneToWorkPlane : CV_PerspectiveTran;
Cam1PlaneToWorkPlane_Execute : BOOL;
Cam1PlaneToWorkPlane_Done : BOOL;
Cam2PlaneToWorkPlane : CV_PerspectiveTran;
Cam2PlaneToWorkPlane_Execute : BOOL;
Cam2PlaneToWorkPlane_Done : BOOL;
Cam1_TagFilterColor1 : IMG_FilterColorHSV;
Cam1_TagFilterColor1_OutWidth : UINT;
Cam1_TagFilterColor1_OutHeight : UINT;
Cam1_TagFilterColor1_OutChannel : UINT;
Cam1_TagFilterColor1_Execute : BOOL;
Cam1_TagFilterColor1_Done : BOOL;
Cam2_TagFilterColor1 : IMG_FilterColorHSV;
Cam2_TagFilterColor1_OutWidth : UINT;
Cam2_TagFilterColor1_OutHeight : UINT;
Cam2_TagFilterColor1_OutChannel : UINT;
Cam2_TagFilterColor1_Execute : BOOL;
Cam2_TagFilterColor1_Done : BOOL;


APP_FindOneTagContours1 : APP_FindOneTagContours;
APP_FindOneTagContours1_Execute : BOOL;
APP_FindOneTagContours1_TypeNum : DINT;
APP_FindOneTagContours1_ErrorDistance : DINT;
APP_FindOneTagContours1_Done : BOOL;
APP_FindOneTagContours1_OutWidth : UINT;
APP_FindOneTagContours1_OutHeight : UINT;
APP_FindOneTagContours1_OutChannel : UINT;
APP_FindOneTagContours1_EN : BOOL;

Cam1_BGFilterColor1 : IMG_FilterColorHSV;
Cam1_BGFilterColor1_Hmax : REAL;
Cam1_BGFilterColor1_Hmin : REAL;
Cam1_BGFilterColor1_Smax : REAL;
Cam1_BGFilterColor1_Smin : REAL;
Cam1_BGFilterColor1_Vmax : REAL;
Cam1_BGFilterColor1_Vmin : REAL;
Cam1_BGFilterColor1_OutWidth : UINT;
Cam1_BGFilterColor1_OutHeight : UINT;
Cam1_BGFilterColor1_OutChannel : UINT;
Cam1_BGFilterColor1_Execute : BOOL;
Cam1_BGFilterColor1_Done : BOOL;


END_VAR

VAR_INPUT

END_VAR

VAR_OUTPUT

END_VAR


IF APP_AdjustPlaneZ1_EN THEN
    APP_AdjustPlaneZ1(Execute := APP_AdjustPlaneZ1_Execute, Plane := ADR(APP_AdjustPlaneZ1_Plane[0]), 
        SrcPoints := ADR(OnPlane3DPoint[0]), Range := ADR(APP_AdjustPlaneZ1_Range[0]), 
        Done =&gt; APP_AdjustPlaneZ1_Done, DstPoints =&gt; APP_AdjustPlaneZ1_DstPoints[0]);
    APP_AdjustPlaneZ1_EN := APP_AdjustPlaneZ1_Execute OR APP_AdjustPlaneZ1_Done;
END_IF;
IF APP_CalBestPlank_ST1_EN THEN
    APP_CalBestPlank_ST1(Execute := APP_CalBestPlank_ST1_Execute, 
        PlansRects := ADR(RealOutContours[1]), PlansNum := PlanNum, LabelsIndex := ADR(TagFlags[0]), 
        LabelNum := InTagNum, MinAreaRatio := APP_CalBestPlank_ST1_MinAreaRatio, 
        Done =&gt; APP_CalBestPlank_ST1_Done, Rect =&gt; BestRect[0], LabelIndex =&gt; MatchLabelIndex, 
        Found =&gt; RectFound, Ratio =&gt; MatchRatio);
    APP_CalBestPlank_ST1_EN := APP_CalBestPlank_ST1_Execute OR APP_CalBestPlank_ST1_Done;
END_IF;
IF APP_ClassfyPolysByTags1_EN THEN
    APP_ClassfyPolysByTags1(Execute := APP_ClassfyPolysByTags1_Execute, 
        Contours := ADR(FixContours[0]), TagsCenters := ADR(HightestTagCenter[0]), 
        TagsVertexs := ADR(HightestTagVertex[0]), TagNum := HightestTagNum, 
        Done =&gt; APP_ClassfyPolysByTags1_Done, OneTagsContoursNum =&gt; OneTagsContoursNum, 
        OneContoursIndex =&gt; OneContoursIndex[0], OneTagsIndexs =&gt; OneTagsIndexs[0], 
        OneTagsCenters =&gt; OneTagsCenters[0], OneTagsVertexs =&gt; OneTagsVertexs[0], 
        DivideTagsContoursNum =&gt; DivideTagsContoursNum, DivideContoursIndex =&gt; DivideContoursIndex[0], 
        DivideTagsNum =&gt; DivideTagsNum[0], DivideTagsIndexs =&gt; DivideTagsIndexs[0], 
        DivideTagsCenters =&gt; DivideTagsCenters[0], DivideTagsVertexs =&gt; DivideTagsVertexs[0], 
        TestDrawTagsContoursNum =&gt; TestDrawTagsContoursNum, 
        TestDrawContoursIndex =&gt; TestDrawContoursIndex[0], TestDrawTagsNum =&gt; TestDrawTagsNum[0], 
        TestDrawTagsIndexs =&gt; TestDrawTagsIndexs[0], TestDrawTagsCenters =&gt; TestDrawTagsCenters[0], 
        TestDrawTagsVertexs =&gt; TestDrawTagsVertexs[0]);
    APP_ClassfyPolysByTags1_EN := APP_ClassfyPolysByTags1_Execute OR APP_ClassfyPolysByTags1_Done;
END_IF;
IF APP_DrawContours1_EN THEN
    APP_DrawContours1(Execute := APP_DrawContours1_Execute, InWidth := Mood_ContoursExtract1_OutWidth, 
        InHeight := Mood_ContoursExtract1_OutHeight, InChannel := Mood_ContoursExtract1_OutChannel, 
        Contours := ADR(LoopFixContour[0]), Color := ADR(APP_DrawContours1_Color[0]), 
        Thickness := APP_DrawContours1_Thickness, Done =&gt; APP_DrawContours1_Done, OutImage =&gt; DstImage9[0], 
        OutWidth =&gt; APP_DrawContours1_OutWidth, OutHeight =&gt; APP_DrawContours1_OutHeight, 
        OutChannel =&gt; APP_DrawContours1_OutChannel);
    APP_DrawContours1_EN := APP_DrawContours1_Execute OR APP_DrawContours1_Done;
END_IF;
IF APP_DrawDivisionRects1_EN THEN
    APP_DrawDivisionRects1(Execute := APP_DrawDivisionRects1_Execute, ImageBuf := ADR(ImageBuf[0,0,0]), 
        InWidth := Mood_ContoursExtract1_OutWidth, InHeight := Mood_ContoursExtract1_OutHeight, 
        InChannel := Mood_ContoursExtract1_OutChannel, RectContours := ADR(OutContours[1]), 
        RectNum := InTagNum, PlanNum := PlanNum, Done =&gt; APP_DrawDivisionRects1_Done);
    APP_DrawDivisionRects1_EN := APP_DrawDivisionRects1_Execute OR APP_DrawDivisionRects1_Done;
END_IF;
IF APP_FilterColorHSV2_EN THEN
    APP_FilterColorHSV2(Execute := APP_FilterColorHSV2_Execute, SrcImage := ADR(DstImage2[0]), 
        Width := Cam1PicWidth, Height := Cam1PicHeight, Channel := Cam1PicChannel, 
        ImageBuf := ADR(ImageBuf[1,0,0]), BkgColorHSV := ADR(Hsv4[0]), 
        MedSize := APP_FilterColorHSV2_MedSize, EroSize := APP_FilterColorHSV2_EroSize, 
        DilSize := APP_FilterColorHSV2_DilSize, Done =&gt; APP_FilterColorHSV2_Done, 
        DstImage =&gt; DstImage10[0]);
    APP_FilterColorHSV2_EN := APP_FilterColorHSV2_Execute OR APP_FilterColorHSV2_Done;
END_IF;
IF APP_FilterColorHSV3_EN THEN
    APP_FilterColorHSV3(Execute := APP_FilterColorHSV3_Execute, SrcImage := ADR(DstImage2[0]), 
        Width := Mood_PerspTransImage4_DstWidth, Height := Mood_PerspTransImage4_DstHeight, 
        Channel := Mood_PerspTransImage4_DstChannel, ImageBuf := ADR(ImageBuf[0,0,0]), 
        BkgColorHSV := ADR(Hsv4[0]), MedSize := APP_FilterColorHSV2_MedSize, 
        EroSize := APP_FilterColorHSV2_EroSize, DilSize := APP_FilterColorHSV2_DilSize, 
        Done =&gt; APP_FilterColorHSV3_Done, DstImage =&gt; DstImage9[0]);
    APP_FilterColorHSV3_EN := APP_FilterColorHSV3_Execute OR APP_FilterColorHSV3_Done;
END_IF;
IF APP_FindConRects2_EN THEN
    APP_FindConRects2(Execute := APP_FindConRects2_Execute, SrcImage := ADR(DstImage10[0]), 
        SrcWidth := Mood_PerspTransImage4_DstWidth, SrcHeight := Mood_PerspTransImage4_DstHeight, 
        SrcChannel := APP_FindConRects2_SrcChannel, Offset := ADR(APP_FindConRects2_Offset[0]), 
        ImageBuf := ADR(ImageBuf[1,0,0]), MinRectSize := ADR(APP_FindRect_TagMinRectSize[0]), 
        ScaleLimit := ADR(APP_FindRect_TagScaleLimit[0]), Done =&gt; APP_FindConRects2_Done, 
        NoConRects =&gt; APP_FindConRects2_NoConRects, CrCenters =&gt; TagCenter1[0], CaCenters =&gt; TagCenter1[0], 
        CaRects =&gt; TagRect1[0], ContNum =&gt; APP_FindConRects2_ContNum);
    APP_FindConRects2_EN := APP_FindConRects2_Execute OR APP_FindConRects2_Done;
END_IF;
IF APP_FindContoursVertex1_EN THEN
    APP_FindContoursVertex1(Execute := APP_FindContoursVertex1_Execute, 
        SrcWidth := Mood_ContoursExtract1_OutWidth, SrcHeight := Mood_ContoursExtract1_OutHeight, 
        AllContours := ADR(AllContours[0]), Epsilon1 := APP_FindContoursVertex1_Epsilon1, 
        Epsilon2 := APP_FindContoursVertex1_Epsilon2, ThetaLimit := APP_FindContoursVertex1_ThetaLimit, 
        Done =&gt; APP_FindContoursVertex1_Done, OutImage =&gt; DstImage11[0], 
        OutWidth =&gt; APP_FindContoursVertex1_OutWidth, OutHeight =&gt; APP_FindContoursVertex1_OutHeight, 
        OutChannel =&gt; APP_FindContoursVertex1_OutChannel, ContoursVertex =&gt; FixContours[0]);
    APP_FindContoursVertex1_EN := APP_FindContoursVertex1_Execute OR APP_FindContoursVertex1_Done;
END_IF;
IF APP_FilterPlankBySize1_EN THEN
    APP_FilterPlankBySize1(Execute := APP_FilterPlankBySize1_Execute, 
        AllRectNum := APP_FilterPlankBySize1_AllRectNum, 
        AllRectSize := ADR(APP_FilterPlankBySize1_AllRectSize[0]), Rect := ADR(SpaceRect[0]), 
        WidthRedundancy := APP_FilterPlankBySize1_WidthRedundancy, 
        HeightRedundancy := APP_FilterPlankBySize1_HeightRedundancy, Done =&gt; APP_FilterPlankBySize1_Done, 
        IsRightClassRect =&gt; IsRightClassRect);
    APP_FilterPlankBySize1_EN := APP_FilterPlankBySize1_Execute OR APP_FilterPlankBySize1_Done;
END_IF;
IF APP_GetMinAreaRect1_EN THEN
    APP_GetMinAreaRect1(Execute := APP_GetMinAreaRect1_Execute, Contours := ADR(LoopFixContour[0]), 
        Done =&gt; APP_GetMinAreaRect1_Done, Rect =&gt; BestRect[0]);
    APP_GetMinAreaRect1_EN := APP_GetMinAreaRect1_Execute OR APP_GetMinAreaRect1_Done;
END_IF;
IF APP_LocateSuckCorner1_EN THEN
    APP_LocateSuckCorner1(Execute := APP_LocateSuckCorner1_Execute, Poly := ADR(LoopFixContour[0]), 
        AngleRange := ADR(APP_LocateSuckCorner1_AngleRange[0]), XAngle := APP_LocateSuckCorner1_XAngle, 
        CornerOffset := ADR(APP_LocateSuckCorner1_CornerOffset[0]), 
        CornerEdgeMinLen := APP_LocateSuckCorner1_CornerEdgeMinLen, Offset := ADR(LocateSuck_Offset[0]), 
        CaLabelsRects := ADR(InTagVertexs[0]), LabelsNum := InTagNum, SuckPadsSize := ADR(TwoSuckSize[0]), 
        SuckPadsNum := APP_LocateSuckCorner1_SuckPadsNum, Done =&gt; APP_LocateSuckCorner1_Done, 
        PadXAngles =&gt; PadXAngle[0], CrRects =&gt; CrRects1[0], PadRCenters =&gt; PadRCenter1[0], 
        SuckPadIndexes =&gt; SuckerIndex1[0], LabelIndexes =&gt; LocateSucker_LabelIndex[0], 
        PlanNum =&gt; APP_LocateSuckCorner1_PlanNum);
    APP_LocateSuckCorner1_EN := APP_LocateSuckCorner1_Execute OR APP_LocateSuckCorner1_Done;
END_IF;
IF APP_LabelsIntersection1_EN THEN
    APP_LabelsIntersection1(Execute := APP_LabelsIntersection1_Execute, IndexA := ADR(MoveTagIndex[0]), 
        NumA := MoveTagNum, IndexB := ADR(MoveTagIndex1[0]), NumB := MoveTagNum1, 
        Done =&gt; APP_LabelsIntersection1_Done, IndexOut =&gt; RealMoveTagIndex[0], NumOut =&gt; RealMoveTagNum);
    APP_LabelsIntersection1_EN := APP_LabelsIntersection1_Execute OR APP_LabelsIntersection1_Done;
END_IF;
IF APP_LocatePullCorner1_EN THEN
    APP_LocatePullCorner1(Execute := APP_LocatePullCorner1_Execute, Poly := ADR(LoopFixContour[0]), 
        AngleRange := ADR(APP_LocatePullCorner1_AngleRange[0]), 
        CornerEdgeMinLen := APP_LocatePullCorner1_CornerEdgeMinLen, 
        Offset := ADR(APP_LocatePullCorner1_Offset[0]), SuckPadsSize := ADR(TwoSuckSize[0]), 
        PullDist := APP_LocatePullCorner1_PullDist, Done =&gt; APP_LocatePullCorner1_Done, 
        PadXAngles =&gt; PadXAngle[0], PadCenters =&gt; PadRCenter1[0], 
        PullVectors =&gt; APP_LocatePullCorner1_PullVectors[0], PlanNum =&gt; APP_LocatePullCorner1_PlanNum);
    APP_LocatePullCorner1_EN := APP_LocatePullCorner1_Execute OR APP_LocatePullCorner1_Done;
END_IF;
IF APP_MaskImage2_EN THEN
    APP_MaskImage2(Execute := APP_MaskImage2_Execute, SrcImage := ADR(Cam1Pic[0]), 
        SrcWidth := Cam1PicWidth, SrcHeight := Cam1PicHeight, SrcChannel := Cam1PicChannel, 
        MaskPoint := ADR(MaskPoint1[0]), MaskFlag := APP_MaskImage2_MaskFlag, Done =&gt; APP_MaskImage2_Done, 
        OutImage =&gt; DstImage1[0], OutWidth =&gt; APP_MaskImage2_OutWidth, 
        OutHeight =&gt; APP_MaskImage2_OutHeight, OutChannel =&gt; APP_MaskImage2_OutChannel);
    APP_MaskImage2_EN := APP_MaskImage2_Execute OR APP_MaskImage2_Done;
END_IF;
IF APP_MoodDivision1_EN THEN
    APP_MoodDivision1(Execute := APP_MoodDivision1_Execute, InContours := ADR(NormContours[0]), 
        TagNum := InTagNum, TagContours := ADR(OutTagVertexs[0]), Radius := APP_MoodDivision1_Radius, 
        Offset := APP_MoodDivision1_Offset, Done =&gt; APP_MoodDivision1_Done, DivisonNum =&gt; PlanNum, 
        OutContours =&gt; OutContours[0]);
    APP_MoodDivision1_EN := APP_MoodDivision1_Execute OR APP_MoodDivision1_Done;
END_IF;
IF APP_PointOffset2_EN THEN
    APP_PointOffset2(Execute := APP_PointOffset2_Execute, InPoints := ADR(BestRect[0]), 
        Num := APP_PointOffset2_Num, Offset := ADR(Offset1[0]), IsSub := APP_PointOffset2_IsSub, 
        Done =&gt; APP_PointOffset2_Done, OutPoints =&gt; SpaceRect[0]);
    APP_PointOffset2_EN := APP_PointOffset2_Execute OR APP_PointOffset2_Done;
END_IF;
IF APP_PointsOffset1_EN THEN
    APP_PointsOffset1(Execute := APP_PointsOffset1_Execute, InPoints := ADR(RBestRect[0]), 
        Num := APP_PointsOffset1_Num, Offset := ADR(Offset1[0]), IsSub := APP_PointsOffset1_IsSub, 
        Done =&gt; APP_PointsOffset1_Done, OutPoints =&gt; SpaceRect[0]);
    APP_PointsOffset1_EN := APP_PointsOffset1_Execute OR APP_PointsOffset1_Done;
END_IF;
IF APP_PointsRotate1_EN THEN
    APP_PointsRotate1(Execute := APP_PointsRotate1_Execute, Num := TagPointNum, 
        InPoints := ADR(InTagVertexs[0]), Center := ADR(Center[0]), Angle := RotateAngle1, 
        Done =&gt; APP_PointsRotate1_Done, OutPoints =&gt; OutTagVertexs[0]);
    APP_PointsRotate1_EN := APP_PointsRotate1_Execute OR APP_PointsRotate1_Done;
END_IF;
IF APP_PointsRotate2_EN THEN
    APP_PointsRotate2(Execute := APP_PointsRotate2_Execute, Num := APP_PointsRotate2_Num, 
        InPoints := ADR(BestRect[0]), Center := ADR(Center[0]), Angle := RotateAngle2, 
        Done =&gt; APP_PointsRotate2_Done, OutPoints =&gt; RBestRect[0]);
    APP_PointsRotate2_EN := APP_PointsRotate2_Execute OR APP_PointsRotate2_Done;
END_IF;
IF APP_ReadImage1_EN THEN
    APP_ReadImage1(Execute := APP_ReadImage1_Execute, Index := APP_ReadImage1_Index, 
        Done =&gt; APP_ReadImage1_Done, OutImage =&gt; Cam1Pic[0], OutWidth =&gt; Cam1PicWidth, 
        OutHeight =&gt; Cam1PicHeight, OutChannel =&gt; Cam1PicChannel);
    APP_ReadImage1_EN := APP_ReadImage1_Execute OR APP_ReadImage1_Done;
END_IF;
IF APP_RectsAverOverlay1_EN THEN
    APP_RectsAverOverlay1(Execute := APP_RectsAverOverlay1_Execute, SrcImage := ADR(ImageBuf[0,0,0]), 
        SrcWidth := Cam1PicWidth, SrcHeight := Cam1PicHeight, 
        Offset := ADR(APP_RectsAverOverlay1_Offset[0]), RectsA := ADR(TagVertexs[0]), NumA := TagNum, 
        RectsB := ADR(TagRect1[0]), NumB := APP_FindConRects2_ContNum, 
        OverlayThres := APP_RectsAverOverlay1_OverlayThres, Done =&gt; APP_RectsAverOverlay1_Done, 
        Vertex =&gt; APP_RectsAverOverlay1_Vertex[0], Center =&gt; APP_RectsAverOverlay1_Center[0], 
        AverOverlay =&gt; HightestTagOverlapRatio, Num =&gt; APP_RectsAverOverlay1_Num);
    APP_RectsAverOverlay1_EN := APP_RectsAverOverlay1_Execute OR APP_RectsAverOverlay1_Done;
END_IF;
IF APP_RectsOverlay1_EN THEN
    APP_RectsOverlay1(Execute := APP_RectsOverlay1_Execute, SrcImage := ADR(ImageBuf[0,0,0]), 
        SrcWidth := Cam1PicWidth, SrcHeight := Cam1PicHeight, Offset := ADR(OffsetTagPlumpness[0]), 
        RectsA := ADR(TagVertexs[0]), NumA := TagNum, RectsB := ADR(TagRect1[0]), 
        NumB := APP_FindConRects2_ContNum, OverlayThres := APP_RectsOverlay1_OverlayThres, 
        MoreOrLess := APP_RectsOverlay1_MoreOrLess, Done =&gt; APP_RectsOverlay1_Done, 
        Vertex =&gt; HightestTagVertex[0], Center =&gt; HightestTagCenter[0], Index =&gt; HightestTagIndex[0], 
        Num =&gt; HightestTagNum);
    APP_RectsOverlay1_EN := APP_RectsOverlay1_Execute OR APP_RectsOverlay1_Done;
END_IF;
IF APP_RectsOverlay3_EN THEN
    APP_RectsOverlay3(Execute := APP_RectsOverlay3_Execute, SrcImage := ADR(ImageBuf[2,0,0]), 
        SrcWidth := Cam1PicWidth, SrcHeight := Cam1PicHeight, Offset := ADR(OffsetTagPlumpness[0]), 
        RectsA := ADR(InTagVertexs[0]), NumA := InTagNum, RectsB := ADR(APP_FindConRects3_CaRects[0]), 
        NumB := APP_FindConRects3_ContNum, OverlayThres := APP_RectsOverlay3_OverlayThres, 
        MoreOrLess := APP_RectsOverlay3_MoreOrLess, Done =&gt; APP_RectsOverlay3_Done, 
        Vertex =&gt; MoveTagVertex[0], Center =&gt; MoveTagCenter[0], Index =&gt; MoveTagIndex[0], 
        Num =&gt; MoveTagNum);
    APP_RectsOverlay3_EN := APP_RectsOverlay3_Execute OR APP_RectsOverlay3_Done;
END_IF;
IF APP_SaveImage1_EN THEN
    APP_SaveImage1(Execute := APP_SaveImage1_Execute, InImage := ADR(Cam1Pic[0]), 
        InWidth := Cam1PicWidth, InHeight := Cam1PicHeight, InChannel := Cam1PicChannel, 
        ImageIndex := APP_SaveImage1_ImageIndex, Done =&gt; APP_SaveImage1_Done, 
        Error =&gt; APP_SaveImage1_Error);
    APP_SaveImage1_EN := APP_SaveImage1_Execute OR APP_SaveImage1_Done;
END_IF;
IF APP_SaveImage3_EN THEN
    APP_SaveImage3(Execute := APP_SaveImage3_Execute, InImage := ADR(DstImage11[0]), 
        InWidth := Cam1PicWidth, InHeight := Cam1PicHeight, InChannel := Cam1PicChannel, 
        ImageIndex := APP_SaveImage3_ImageIndex, Done =&gt; APP_SaveImage3_Done, 
        Error =&gt; APP_SaveImage3_Error);
    APP_SaveImage3_EN := APP_SaveImage3_Execute OR APP_SaveImage3_Done;
END_IF;
IF APP_SaveImage4_EN THEN
    APP_SaveImage4(Execute := APP_SaveImage4_Execute, InImage := ADR(ImageBuf[0,0,0]), 
        InWidth := Cam1PicWidth, InHeight := Cam1PicHeight, InChannel := APP_SaveImage4_InChannel, 
        ImageIndex := APP_SaveImage4_ImageIndex, Done =&gt; APP_SaveImage4_Done, 
        Error =&gt; APP_SaveImage4_Error);
    APP_SaveImage4_EN := APP_SaveImage4_Execute OR APP_SaveImage4_Done;
END_IF;
IF APP_SaveImage5_EN THEN
    APP_SaveImage5(Execute := APP_SaveImage5_Execute, InImage := ADR(DstImage2[0]), 
        InWidth := Cam1PicWidth, InHeight := Cam1PicHeight, InChannel := APP_SaveImage5_InChannel, 
        ImageIndex := APP_SaveImage5_ImageIndex, Done =&gt; APP_SaveImage5_Done, 
        Error =&gt; APP_SaveImage5_Error);
    APP_SaveImage5_EN := APP_SaveImage5_Execute OR APP_SaveImage5_Done;
END_IF;
IF APP_SetRightContours1_EN THEN
    APP_SetRightContours1(Execute := APP_SetRightContours1_Execute, 
        InContours := ADR(LoopFixContour[0]), InWidth := Mood_ContoursExtract1_OutWidth, 
        InHeight := Mood_ContoursExtract1_OutHeight, Done =&gt; APP_SetRightContours1_Done, 
        OutImage =&gt; DstImage1[0], OutWidth =&gt; APP_SetRightContours1_OutWidth, 
        OutHeight =&gt; APP_SetRightContours1_OutHeight, OutChannel =&gt; APP_SetRightContours1_OutChannel, 
        OutContours =&gt; NormContours[0], Angle =&gt; RotateAngle1);
    APP_SetRightContours1_EN := APP_SetRightContours1_Execute OR APP_SetRightContours1_Done;
END_IF;
IF APP_ThresholdImg1_EN THEN
    APP_ThresholdImg1(Execute := APP_ThresholdImg1_Execute, InWidth := Mood_ContoursExtract1_OutWidth, 
        InHeight := Mood_ContoursExtract1_OutHeight, InChannel := Mood_ContoursExtract1_OutChannel, 
        Threshold := APP_ThresholdImg_Threshold, Value := APP_ThresholdImg_Value, 
        Done =&gt; APP_ThresholdImg1_Done, OutImage =&gt; DstImage9[0], OutWidth =&gt; APP_ThresholdImg_OutWidth, 
        OutHeight =&gt; APP_ThresholdImg_OutHeight, OutChannel =&gt; APP_ThresholdImg_OutChannel);
    APP_ThresholdImg1_EN := APP_ThresholdImg1_Execute OR APP_ThresholdImg1_Done;
END_IF;

IF APP_TransToolAbsPos2D1_EN THEN
    APP_TransToolAbsPos2D1(Execute := APP_TransToolAbsPos2D1_Execute, 
        RPosition := ADR(APP_FindCenterByRay1_CenterR[0]), Offset := ADR(CornerPointOffset[0]), 
        ToolXOffs := APP_TransToolAbsPos2D1_ToolXOffs, AngleToXPAxis := RotateAngle1, 
        RZOfXPAxis := APP_TransToolAbsPos2D1_RZOfXPAxis, ToolPosition =&gt; ARM2_PTPMoveAbsTargetTCP[1], 
        Done =&gt; APP_TransToolAbsPos2D1_Done);
    APP_TransToolAbsPos2D1_EN := APP_TransToolAbsPos2D1_Execute OR APP_TransToolAbsPos2D1_Done;
END_IF;



IF APP_TransToolAbsPos2D2_EN THEN
    APP_TransToolAbsPos2D2(Execute := APP_TransToolAbsPos2D2_Execute, RPosition := ADR(PadRCenter1[0]), 
        Offset := ADR(CornerPointOffset[0]), ToolXOffs := APP_TransToolAbsPos2D2_ToolXOffs, 
        AngleToXPAxis := RotateAngle2, RZOfXPAxis := APP_TransToolAbsPos2D2_RZOfXPAxis, 
        ToolPosition =&gt; ARM2_PTPMoveAbsTargetTCP[1], Done =&gt; APP_TransToolAbsPos2D2_Done);
    APP_TransToolAbsPos2D2_EN := APP_TransToolAbsPos2D2_Execute OR APP_TransToolAbsPos2D2_Done;
END_IF;


IF APP_UINT2REAL1_EN THEN
    APP_UINT2REAL1(Execute := APP_UINT2REAL1_Execute, InArray := ADR(OutContours[0]), 
        Len := OutContoursLength, Done =&gt; APP_UINT2REAL1_Done, OutArray =&gt; RealOutContours[0]);
    APP_UINT2REAL1_EN := APP_UINT2REAL1_Execute OR APP_UINT2REAL1_Done;
END_IF;
IF CV_PolyMask1_Execute AND NOT CV_PolyMask1_Done THEN
    CV_PolyMask1(InImage := ADR(DstImage2[0]), InWidth := Cam1PicWidth, InHeight := Cam1PicHeight, 
        InChannel := CV_PolyMask1_InChannel, Color := ADR(CV_PolyMask1_Color[0]), 
        PointList := ADR(CV_PolyMask1_PointList[0]), Flag := CV_PolyMask1_Flag, Fill := CV_PolyMask1_Fill, 
        OutImage =&gt; DstImage2[0], OutWidth =&gt; CV_PolyMask1_OutWidth, OutHeight =&gt; CV_PolyMask1_OutHeight, 
        OutChannel =&gt; CV_PolyMask1_OutChannel);
    CV_PolyMask1_Done := TRUE;
END_IF;
IF NOT CV_PolyMask1_Execute THEN
    CV_PolyMask1_Done := FALSE;
END_IF;
IF GRPH_RectInfo1_EN THEN
    GRPH_RectInfo1(Execute := GRPH_RectInfo1_Execute, RectContours := ADR(SpaceRect[0]), 
        Done =&gt; GRPH_RectInfo1_Done, RectInfo =&gt; BestRectInfo[0]);
    GRPH_RectInfo1_EN := GRPH_RectInfo1_Execute OR GRPH_RectInfo1_Done;
END_IF;
IF Mood_ContoursExtract1_EN THEN
    Mood_ContoursExtract1(Execute := Mood_ContoursExtract1_Execute, SrcImage := ADR(DstImage2[0]), 
        SrcImage1 := ADR(DstImage4[0]), ImageBuff := ADR(ImageBuf[0,0,0]), SrcWidth := Cam1PicWidth, 
        SrcHeight := Cam1PicHeight, SrcChannel := Mood_ContoursExtract1_SrcChannel, 
        Points := ADR(MaskPoints[0]), MinContoursArea := Mood_ContoursExtract1_MinContoursArea, 
        Colors := ADR(Color[0]), Flag := Mood_ContoursExtract1_Flag, Fill := Mood_ContoursExtract1_Fill, 
        ErodeKernelSize := Mood_ContoursExtract1_ErodeKernelSize, Done =&gt; Mood_ContoursExtract1_Done, 
        OutImage =&gt; DstImage5[0], OutWidth =&gt; Mood_ContoursExtract1_OutWidth, 
        OutHeight =&gt; Mood_ContoursExtract1_OutHeight, OutChannel =&gt; Mood_ContoursExtract1_OutChannel, 
        RectInfo =&gt; RectInfo[0], AllContours =&gt; AllContours[0]);
    Mood_ContoursExtract1_EN := Mood_ContoursExtract1_Execute OR Mood_ContoursExtract1_Done;
END_IF;
IF Mood_EdgeExtract1_EN THEN
    Mood_EdgeExtract1(Execute := Mood_EdgeExtract1_Execute, SrcImage := ADR(Cam1Pic[0]), 
        SrcWidth := Cam1PicWidth, SrcHeight := Cam1PicHeight, SrcChannel := Cam1PicChannel, 
        CannyThresholdH := Mood_EdgeExtract1_CannyThresholdH, 
        CannyThresholdL := Mood_EdgeExtract1_CannyThresholdL, FilterHSV := ADR(Hsv1[0]), 
        CannyKernelSize := Mood_EdgeExtract1_CannyKernelSize, 
        DilateKernelSize := Mood_EdgeExtract1_DilateKernelSize, 
        ErodeKernelSize := Mood_EdgeExtract1_ErodeKernelSize, Done =&gt; Mood_EdgeExtract1_Done, 
        OutImage =&gt; DstImage1[0], OutWidth =&gt; Mood_EdgeExtract1_OutWidth, 
        OutHeight =&gt; Mood_EdgeExtract1_OutHeight, OutChannel =&gt; Mood_EdgeExtract1_OutChannel);
    Mood_EdgeExtract1_EN := Mood_EdgeExtract1_Execute OR Mood_EdgeExtract1_Done;
END_IF;
IF Mood_EdgeExtract2_EN THEN
    Mood_EdgeExtract2(Execute := Mood_EdgeExtract2_Execute, SrcImage := ADR(Cam2Pic[0]), 
        SrcWidth := Cam2PicWidth, SrcHeight := Cam2PicHeight, SrcChannel := Cam2PicChannel, 
        CannyThresholdH := Mood_EdgeExtract2_CannyThresholdH, 
        CannyThresholdL := Mood_EdgeExtract2_CannyThresholdL, FilterHSV := ADR(Hsv2[0]), 
        CannyKernelSize := Mood_EdgeExtract2_CannyKernelSize, 
        DilateKernelSize := Mood_EdgeExtract2_DilateKernelSize, 
        ErodeKernelSize := Mood_EdgeExtract2_ErodeKernelSize, Done =&gt; Mood_EdgeExtract2_Done, 
        OutImage =&gt; DstImage3[0], OutWidth =&gt; Mood_EdgeExtract2_OutWidth, 
        OutHeight =&gt; Mood_EdgeExtract2_OutHeight, OutChannel =&gt; Mood_EdgeExtract2_OutChannel);
    Mood_EdgeExtract2_EN := Mood_EdgeExtract2_Execute OR Mood_EdgeExtract2_Done;
END_IF;
IF Mood_PerspTransImage1_EN THEN
    Mood_PerspTransImage1(Execute := Mood_PerspTransImage1_Execute, SrcImage := ADR(DstImage1[0]), 
        SrcWidth := Mood_EdgeExtract1_OutWidth, SrcHeight := Mood_EdgeExtract1_OutHeight, 
        SrcChannel := Mood_EdgeExtract1_OutChannel, Offset := ADR(Cam1PicOffset[0]), 
        CaSrcPoints := ADR(Cam1Point2dLowHeight[0]), RDstPoints := ADR(Space3d[0]), 
        Done =&gt; Mood_PerspTransImage1_Done, DstImage =&gt; DstImage2[0], 
        DstWidth =&gt; Mood_PerspTransImage1_DstWidth, DstHeight =&gt; Mood_PerspTransImage1_DstHeight, 
        DstChannel =&gt; Mood_PerspTransImage1_DstChannel, OffsetX =&gt; Cam1OffsetX, OffsetY =&gt; Cam1OffsetY);
    Mood_PerspTransImage1_EN := Mood_PerspTransImage1_Execute OR Mood_PerspTransImage1_Done;
END_IF;
IF Mood_PerspTransImage4_EN THEN
    Mood_PerspTransImage4(Execute := Mood_PerspTransImage4_Execute, SrcImage := ADR(DstImage1[0]), 
        SrcWidth := APP_MaskImage2_OutWidth, SrcHeight := APP_MaskImage2_OutHeight, 
        SrcChannel := APP_MaskImage2_OutChannel, Offset := ADR(Cam1PicOffset[0]), 
        CaSrcPoints := ADR(Cam1Point2d[0]), RDstPoints := ADR(Space3d[0]), 
        Done =&gt; Mood_PerspTransImage4_Done, DstImage =&gt; DstImage2[0], 
        DstWidth =&gt; Mood_PerspTransImage4_DstWidth, DstHeight =&gt; Mood_PerspTransImage4_DstHeight, 
        DstChannel =&gt; Mood_PerspTransImage4_DstChannel, OffsetX =&gt; Cam1OffsetX, OffsetY =&gt; Cam1OffsetY);
    Mood_PerspTransImage4_EN := Mood_PerspTransImage4_Execute OR Mood_PerspTransImage4_Done;
END_IF;
IF Mood_PerspTransImage5_EN THEN
    Mood_PerspTransImage5(Execute := Mood_PerspTransImage5_Execute, SrcImage := ADR(DstImage1[0]), 
        SrcWidth := Cam1PicWidth, SrcHeight := Cam1PicHeight, SrcChannel := Cam1PicChannel, 
        Offset := ADR(Cam1PicOffset[0]), CaSrcPoints := ADR(Cam1Point2dLowHeight[0]), 
        RDstPoints := ADR(Space3d[0]), Done =&gt; Mood_PerspTransImage5_Done, DstImage =&gt; DstImage2[0], 
        DstWidth =&gt; Mood_PerspTransImage5_DstWidth, DstHeight =&gt; Mood_PerspTransImage5_DstHeight, 
        DstChannel =&gt; Mood_PerspTransImage5_DstChannel, OffsetX =&gt; Mood_PerspTransImage5_OffsetX, 
        OffsetY =&gt; Mood_PerspTransImage5_OffsetY);
    Mood_PerspTransImage5_EN := Mood_PerspTransImage5_Execute OR Mood_PerspTransImage5_Done;
END_IF;
IF Mood_PerspTransImage7_EN THEN
    Mood_PerspTransImage7(Execute := Mood_PerspTransImage7_Execute, SrcImage := ADR(DstImage1[0]), 
        SrcWidth := Cam1PicWidth, SrcHeight := Cam1PicHeight, 
        SrcChannel := Mood_PerspTransImage7_SrcChannel, Offset := ADR(Cam1PicOffset[0]), 
        CaSrcPoints := ADR(Cam1Point2d[0]), RDstPoints := ADR(Space3d[0]), 
        Done =&gt; Mood_PerspTransImage7_Done, DstImage =&gt; DstImage2[0], 
        DstWidth =&gt; Mood_PerspTransImage7_DstWidth, DstHeight =&gt; Mood_PerspTransImage7_DstHeight, 
        DstChannel =&gt; Mood_PerspTransImage7_DstChannel, OffsetX =&gt; Mood_PerspTransImage7_OffsetX, 
        OffsetY =&gt; Mood_PerspTransImage7_OffsetY);
    Mood_PerspTransImage7_EN := Mood_PerspTransImage7_Execute OR Mood_PerspTransImage7_Done;
END_IF;
IF Temp_ScaleParallelogram1_EN THEN
    Temp_ScaleParallelogram1(Execute := Temp_ScaleParallelogram1_Execute, 
        InRotateRect := ADR(CV_PolyMask1_PointList[1]), 
        WidthParameter := Temp_ScaleParallelogram1_WidthParameter, 
        HeightParameter := Temp_ScaleParallelogram1_HeightParameter, 
        IsScale := Temp_ScaleParallelogram1_IsScale, Done =&gt; Temp_ScaleParallelogram1_Done, 
        OutRotateRect =&gt; CV_PolyMask1_PointList[1]);
    Temp_ScaleParallelogram1_EN := Temp_ScaleParallelogram1_Execute OR Temp_ScaleParallelogram1_Done;
END_IF;

IF IMG_Resize1_Execute AND NOT IMG_Resize1_Done THEN
    IMG_Resize1(InImage := ADR(Cam1Pic[0]), InWidth := Cam1PicWidth, InHeight := Cam1PicHeight, 
        InChannel := Cam2PicChannel, ResizeW := IMG_Resize1_ResizeW, ResizeH := IMG_Resize1_ResizeH, 
        OutImage =&gt; IMG_Resize1_OutImage[0], OutWidth =&gt; IMG_Resize1_OutWidth, 
        OutHeight =&gt; IMG_Resize1_OutHeight, OutChannel =&gt; IMG_Resize1_OutChannel);
    IMG_Resize1_Done := TRUE;
END_IF;
IF NOT IMG_Resize1_Execute THEN
    IMG_Resize1_Done := FALSE;
END_IF;
IF IMG_Resize2_Execute AND NOT IMG_Resize2_Done THEN
    IMG_Resize2(InImage := ADR(Cam2Pic[0]), InWidth := Cam2PicWidth, InHeight := Cam2PicHeight, 
        InChannel := Cam2PicChannel, ResizeW := IMG_Resize2_ResizeW, ResizeH := IMG_Resize2_ResizeH, 
        OutImage =&gt; IMG_Resize2_OutImage[0], OutWidth =&gt; IMG_Resize2_OutWidth, 
        OutHeight =&gt; IMG_Resize2_OutHeight, OutChannel =&gt; IMG_Resize2_OutChannel);
    IMG_Resize2_Done := TRUE;
END_IF;
IF NOT IMG_Resize2_Execute THEN
    IMG_Resize2_Done := FALSE;
END_IF;

IF CV_Canny1_Execute AND NOT CV_Canny1_Done THEN
    CV_Canny1(InImage := ADR(Cam1Pic[0]), InWidth := Cam1PicWidth, InHeight := Cam1PicHeight, 
        InChannel := Cam1PicChannel, ThresholdH := 200, ThresholdL := 100, KernelSize := 3, 
        OutImage =&gt; CV_Canny1_OutImage[0], OutWidth =&gt; CV_Canny1_OutWidth, 
        OutHeight =&gt; CV_Canny1_OutHeight, OutChannel =&gt; CV_Canny1_OutChannel);
    CV_Canny1_Done := TRUE;
END_IF;
IF NOT CV_Canny1_Execute THEN
    CV_Canny1_Done := FALSE;
END_IF;



IF APP_FindCenterByRay1_EN THEN
    APP_FindCenterByRay1(Execute := APP_FindCenterByRay1_Execute, SrcImage := ADR(EdgeAddImage[0]), 
        SrcWidth := Cam1_Perspective1_OutWidth, SrcHeight := Cam1_Perspective1_OutHeight, 
        SrcChannel := Cam1_Perspective1_OutChannel, AllContours := ADR(AllContours[0]), 
        TopTagCenters := ADR(HightestTagCenter[0]), Vertex := ADR(dintArray[0]), VerNum := HightestTagNum, 
        RectType := ADR(APP_FindCenterByRay1_RectType[0]), TypeNum := APP_FindCenterByRay1_TypeNum, 
        ErrorDistance := APP_FindCenterByRay1_ErrorDistance, 
        PullTargetPoint := ADR(APP_FindCenterByRay1_PullTargetPoint[0]), 
        OffsetX := RobotToWorkSpaceOffsetX, OffsetY := RobotToWorkSpaceOffsetY, 
        Epsilon1 := APP_FindCenterByRay1_Epsilon1, Epsilon2 := APP_FindCenterByRay1_Epsilon2, 
        ThetaLimit := APP_FindCenterByRay1_ThetaLimit, Done =&gt; APP_FindCenterByRay1_Done, 
        OutImage =&gt; ImageBuf[0,0,0], OutWidth =&gt; APP_FindCenterByRay1_OutWidth, 
        OutHeight =&gt; APP_FindCenterByRay1_OutHeight, OutChannel =&gt; APP_FindCenterByRay1_OutChannel, 
        ContoursVertex =&gt; APP_FindCenterByRay1_ContoursVertex[0], OptionalRects =&gt; OutContours[0], 
        CenterR =&gt; APP_FindCenterByRay1_CenterR[0], PullPoint =&gt; PadRCenter1[0], 
        RectWidth =&gt; BestRectInfo[2], RectHeight =&gt; BestRectInfo[3], Angle =&gt; RotateAngle1, 
        checkFlag =&gt; AlreadyGetPlank);
    APP_FindCenterByRay1_EN := APP_FindCenterByRay1_Execute OR APP_FindCenterByRay1_Done;
END_IF;


IF APP_ContourClosetSort1_EN THEN
    APP_ContourClosetSort1(Execute := APP_ContourClosetSort1_Execute, Contours := ADR(OutContours[0]), 
        PointIn := ADR(APP_ContourClosetSort_PointIn[0]), Done =&gt; APP_ContourClosetSort1_Done, 
        ContoursOut =&gt; SortedContours[0], Dist2Arr =&gt; APP_ContourClosetSort_Dist2Arr[0], 
        Dist2Idx =&gt; APP_ContourClosetSort_Dist2Idx[0]);
    APP_ContourClosetSort1_EN := APP_ContourClosetSort1_Execute OR APP_ContourClosetSort1_Done;
END_IF;

IF Cam1_FindTagContours1_EN THEN
    Cam1_FindTagContours1(Execute := Cam1_FindTagContours1_Execute, 
        SrcImage := ADR(Cam1_TagFilterColor1_OutImage[0]), SrcWidth := Cam1_Perspective1_OutWidth, 
        SrcHeight := Cam1_Perspective1_OutHeight, SrcChannel := Cam1_FindTagContours1_SrcChannel, 
        Offset := ADR(Cam1_FindTagContours1_Offset[0]), ImageBuf := ADR(Cam1ImageTemp1[0]), 
        MinRectSize := ADR(APP_FindRect_TagMinRectSize[0]), 
        ScaleLimit := ADR(APP_FindRect_TagScaleLimit[0]), Done =&gt; Cam1_FindTagContours1_Done, 
        NoConRects =&gt; Cam1_FindTagContours1_NoConRects, CrCenters =&gt; Cam1_FindTagCenters[0], 
        CaCenters =&gt; Cam1_FindTagCenters[0], CaRects =&gt; Cam1_FindTagRects[0], ContNum =&gt; Cam1_FindTagtNum);
    Cam1_FindTagContours1_EN := Cam1_FindTagContours1_Execute OR Cam1_FindTagContours1_Done;
END_IF;
IF Cam1_CannyEdge_Execute AND NOT Cam1_CannyEdge_Done THEN
    Cam1_CannyEdge(InImage := ADR(Cam1_Perspective1_OutImage[0]), 
        InWidth := Cam1_Perspective1_OutWidth, InHeight := Cam1_Perspective1_OutHeight, 
        InChannel := Cam1_Perspective1_OutChannel, ThresholdH := Cam1_CannyEdge_ThresholdH, 
        ThresholdL := Cam1_CannyEdge_ThresholdL, KernelSize := Cam1_CannyEdge_KernelSize, 
        OutImage =&gt; Cam1_CannyEdge_OutImage[0], OutWidth =&gt; Cam1_CannyEdge_OutWidth, 
        OutHeight =&gt; Cam1_CannyEdge_OutHeight, OutChannel =&gt; Cam1_CannyEdge_OutChannel);
    Cam1_CannyEdge_Done := TRUE;
END_IF;
IF NOT Cam1_CannyEdge_Execute THEN
    Cam1_CannyEdge_Done := FALSE;
END_IF;


IF Cam1PlaneToWorkPlane_Execute AND NOT Cam1PlaneToWorkPlane_Done THEN
    Cam1PlaneToWorkPlane(RectPoint := ADR(Cam1RoiPoint2d[0]), 
        RectPoint2 := ADR(WorkSpaceFourPointsXY[0]), TransRect =&gt; Cam1PlaneToWorkPlaneMatrix1[0]);
    Cam1PlaneToWorkPlane_Done := TRUE;
END_IF;
IF NOT Cam1PlaneToWorkPlane_Execute THEN
    Cam1PlaneToWorkPlane_Done := FALSE;
END_IF;
IF Cam1_Perspective1_Execute AND NOT Cam1_Perspective1_Done THEN
    Cam1_Perspective1(InImage := ADR(Cam1Pic[0]), InWidth := Cam1PicWidth, InHeight := Cam1PicHeight, 
        InChannel := Cam1PicChannel, Matrix := ADR(Cam1PlaneToWorkPlaneMatrix1[0]), 
        OutMaxWidth := WorkSpaceWidth, OutMaxHeight := WorkSpaceHeight, 
        OutImage =&gt; Cam1_Perspective1_OutImage[0], OutWidth =&gt; Cam1_Perspective1_OutWidth, 
        OutHeight =&gt; Cam1_Perspective1_OutHeight, OutChannel =&gt; Cam1_Perspective1_OutChannel);
    Cam1_Perspective1_Done := TRUE;
END_IF;
IF NOT Cam1_Perspective1_Execute THEN
    Cam1_Perspective1_Done := FALSE;
END_IF;
IF Cam1_TagFilterColor1_Execute AND NOT Cam1_TagFilterColor1_Done THEN
    Cam1_TagFilterColor1(InImage := ADR(Cam1_Perspective1_OutImage[0]), 
        InWidth := Cam1_Perspective1_OutWidth, InHeight := Cam1_Perspective1_OutHeight, 
        InChannel := Cam1_Perspective1_OutChannel, Hmax := Cam1_TagHsv1[0], Hmin := Cam1_TagHsv1[1], 
        Smax := Cam1_TagHsv1[2], Smin := Cam1_TagHsv1[3], Vmax := Cam1_TagHsv1[4], Vmin := Cam1_TagHsv1[5], 
        OutImage =&gt; Cam1_TagFilterColor1_OutImage[0], OutWidth =&gt; Cam1_TagFilterColor1_OutWidth, 
        OutHeight =&gt; Cam1_TagFilterColor1_OutHeight, OutChannel =&gt; Cam1_TagFilterColor1_OutChannel);
    Cam1_TagFilterColor1_Done := TRUE;
END_IF;
IF NOT Cam1_TagFilterColor1_Execute THEN
    Cam1_TagFilterColor1_Done := FALSE;
END_IF;

IF Cam1_BGFilterColor1_Execute AND NOT Cam1_BGFilterColor1_Done THEN
    Cam1_BGFilterColor1(InImage := ADR(Cam1_Perspective1_OutImage[0]), 
        InWidth := Cam1_Perspective1_OutWidth, InHeight := Cam1_Perspective1_OutHeight, 
        InChannel := Cam1_Perspective1_OutChannel, Hmax := Cam1_BGFilterColor1_Hmax, 
        Hmin := Cam1_BGFilterColor1_Hmin, Smax := Cam1_BGFilterColor1_Smax, 
        Smin := Cam1_BGFilterColor1_Smin, Vmax := Cam1_BGFilterColor1_Vmax, 
        Vmin := Cam1_BGFilterColor1_Vmin, OutImage =&gt; Cam1_BGFilterColor1_OutImage[0], 
        OutWidth =&gt; Cam1_BGFilterColor1_OutWidth, OutHeight =&gt; Cam1_BGFilterColor1_OutHeight, 
        OutChannel =&gt; Cam1_BGFilterColor1_OutChannel);
    Cam1_BGFilterColor1_Done := TRUE;
END_IF;
IF NOT Cam1_BGFilterColor1_Execute THEN
    Cam1_BGFilterColor1_Done := FALSE;
END_IF;

IF APP_RectsAverOverlay_Tag_EN THEN
    APP_RectsAverOverlay_Tag(Execute := APP_RectsAverOverlay_Tag_Execute, 
        SrcImage := ADR(ImageTemp1[0]), SrcWidth := Cam1_Perspective1_OutWidth, 
        SrcHeight := Cam1_Perspective1_OutHeight, Offset := ADR(APP_RectsAverOverlay_Tag_Offset[0]), 
        RectsA := ADR(Cam1_FindTagRects[0]), NumA := Cam1_FindTagtNum, RectsB := ADR(Cam2_FindTagRects[0]), 
        NumB := Cam2_FindTagtNum, OverlayThres := HightestTagOverlapRatioThreshold, 
        Done =&gt; APP_RectsAverOverlay_Tag_Done, Vertex =&gt; HightestTagVertex[0], 
        Center =&gt; HightestTagCenter[0], AverOverlay =&gt; HightestTagOverlapRatio, Num =&gt; HightestTagNum);
    APP_RectsAverOverlay_Tag_EN := APP_RectsAverOverlay_Tag_Execute OR APP_RectsAverOverlay_Tag_Done;
END_IF;
IF Cam1And2_EdgeAdd_Execute AND NOT Cam1And2_EdgeAdd_Done THEN
    Cam1And2_EdgeAdd(InImage := ADR(Cam1_CannyEdge_OutImage[0]), InWidth := Cam1_Perspective1_OutWidth, 
        InHeight := Cam1_Perspective1_OutHeight, InChannel := Cam1And2_EdgeAdd_InChannel, 
        InImage2 := ADR(Cam2_CannyEdge_OutImage[0]), OutImage =&gt; EdgeAddImage[0], 
        OutWidth =&gt; Cam1And2_EdgeAdd_OutWidth, OutHeight =&gt; Cam1And2_EdgeAdd_OutHeight, 
        OutChannel =&gt; Cam1And2_EdgeAdd_OutChannel);
    Cam1And2_EdgeAdd_Done := TRUE;
END_IF;
IF NOT Cam1And2_EdgeAdd_Execute THEN
    Cam1And2_EdgeAdd_Done := FALSE;
END_IF;
IF CV_EdgeAddDilate_Execute AND NOT CV_EdgeAddDilate_Done THEN
    CV_EdgeAddDilate(InImage := ADR(EdgeAddImage[0]), InWidth := Cam1_Perspective1_OutWidth, 
        InHeight := Cam1_Perspective1_OutHeight, InChannel := CV_EdgeAddDilate_InChannel, 
        KerSizeW := CV_EdgeAddDilate_KerSizeW, KerSizeH := CV_EdgeAddDilate_KerSizeW, 
        OutImage =&gt; EdgeAddImage[0], OutWidth =&gt; CV_EdgeAddDilate_OutWidth, 
        OutHeight =&gt; CV_EdgeAddDilate_OutHeight, OutChannel =&gt; CV_EdgeAddDilate_OutChannel);
    CV_EdgeAddDilate_Done := TRUE;
END_IF;
IF NOT CV_EdgeAddDilate_Execute THEN
    CV_EdgeAddDilate_Done := FALSE;
END_IF;
IF CV_FindPlankContours_Execute AND NOT CV_FindPlankContours_Done THEN
    CV_FindPlankContours(InImage := ADR(EdgeAddImage[0]), InWidth := Cam1_Perspective1_OutWidth, 
        InHeight := Cam1_Perspective1_OutHeight, InChannel := CV_FindPlankContours_InChannel, 
        MinContoursArea := MinPlankContoursArea, OutImage =&gt; CV_FindPlankContours_OutImage[0], 
        OutWidth =&gt; CV_FindPlankContours_OutWidth, OutHeight =&gt; CV_FindPlankContours_OutHeight, 
        OutChannel =&gt; CV_FindPlankContours_OutChannel, RectPoint =&gt; CV_FindPlankContours_RectPoint[0], 
        AllContours =&gt; AllContours[0]);
    CV_FindPlankContours_Done := TRUE;
END_IF;
IF NOT CV_FindPlankContours_Execute THEN
    CV_FindPlankContours_Done := FALSE;
END_IF;


IF APP_FindOneTagContours1_EN THEN
    APP_FindOneTagContours1(Execute := APP_FindOneTagContours1_Execute, 
        SrcImage := ADR(EdgeAddImage[0]), SrcWidth := Cam1_Perspective1_OutWidth, 
        SrcHeight := Cam1_Perspective1_OutHeight, SrcChannel := Cam1_Perspective1_OutChannel, 
        AllContours := ADR(AllContours[0]), TagsVertexs := ADR(HightestTagVertex[0]), 
        TagNum := HightestTagNum, RectType := ADR(APP_FindCenterByRay1_RectType[0]), 
        TypeNum := APP_FindOneTagContours1_TypeNum, ErrorDistance := APP_FindOneTagContours1_ErrorDistance, 
        OffsetX := RobotToWorkSpaceOffsetX, OffsetY := RobotToWorkSpaceOffsetY, 
        Done =&gt; APP_FindOneTagContours1_Done, OutImage =&gt; ImageBuf[0,0,0], 
        OutWidth =&gt; APP_FindOneTagContours1_OutWidth, OutHeight =&gt; APP_FindOneTagContours1_OutHeight, 
        OutChannel =&gt; APP_FindOneTagContours1_OutChannel, OptionalRects =&gt; OutContours[0], 
        CenterR =&gt; APP_FindCenterByRay1_CenterR[0], RectWidth =&gt; BestRectInfo[2], 
        RectHeight =&gt; BestRectInfo[3], Angle =&gt; RotateAngle1, checkFlag =&gt; AlreadyGetPlank);
    APP_FindOneTagContours1_EN := APP_FindOneTagContours1_Execute OR APP_FindOneTagContours1_Done;
END_IF;

END_PROGRAM