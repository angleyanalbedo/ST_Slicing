PROGRAM MC_MoveCircularRelative

VAR
    FB_ACS_MCS_KinTransform1 : FB_ACS_MCS_KinTransform;
    FB_MCS_ACS_KinTransform1 : FB_MCS_ACS_KinTransform;
        mcs_acs_err : BOOL;
    FB_CircularPointsCal1 : FB_CircularPointsCal;
        nowpoint_tcp : ARRAY[1..6] OF REAL;
        strpoint_tcp : ARRAY[1..6] OF REAL;
    FB_CircularPointBufCal1 : FB_CircularPointBufCal;
        cirerror : BOOL;
        cirerrrorid : UINT;
        center_point : ARRAY[1..3] OF REAL;
        circle_path : ARRAY[1..600] OF REAL;
        path_s : ARRAY[1..100] OF REAL;    // 相对于起始点
        path_point_num : USINT;
        path_point_start : USINT;
        path_point_end : USINT;
    MC_PositionProfile1 : MC_PositionProfile;
    MC_PositionProfile2 : MC_PositionProfile;
    MC_PositionProfile3 : MC_PositionProfile;
    MC_PositionProfile4 : MC_PositionProfile;
    MC_PositionProfile5 : MC_PositionProfile;
    MC_PositionProfile6 : MC_PositionProfile;
    MC_PositionProfile7 : MC_PositionProfile;
    MC_PositionProfile8 : MC_PositionProfile;
        pointbuf_pus1 : ARRAY[1..100] OF REAL;
        pointbuf_pus2 : ARRAY[1..100] OF REAL;
        pointbuf_pus3 : ARRAY[1..100] OF REAL;
        pointbuf_pus4 : ARRAY[1..100] OF REAL;
        pointbuf_pus5 : ARRAY[1..100] OF REAL;
        pointbuf_pus6 : ARRAY[1..100] OF REAL;
        pointbuf_pus7 : ARRAY[1..100] OF REAL;
        pointbuf_pus8 : ARRAY[1..100] OF REAL;
        pp_exe : ARRAY[1..8] OF BOOL;
        pp_done : BOOL;
        pp_don : ARRAY[1..8] OF BOOL;
        pp_don1 : BOOL; pp_don2 : BOOL; pp_don3 : BOOL; pp_don4 : BOOL; pp_don5 : BOOL; pp_don6 : BOOL; pp_don7 : BOOL; pp_don8 : BOOL;
        pp_busy : BOOL;
        pp_bus : ARRAY[1..8] OF BOOL;
        pp_bus1 : BOOL; pp_bus2 : BOOL; pp_bus3 : BOOL; pp_bus4 : BOOL; pp_bus5 : BOOL; pp_bus6 : BOOL; pp_bus7 : BOOL; pp_bus8 : BOOL;
        pp_active : BOOL;
        pp_active_last : BOOL;
        pp_act : ARRAY[1..8] OF BOOL;
        pp_act1 : BOOL; pp_act2 : BOOL; pp_act3 : BOOL; pp_act4 : BOOL; pp_act5 : BOOL; pp_act6 : BOOL; pp_act7 : BOOL; pp_act8 : BOOL;
        pp_abr : ARRAY[1..8] OF BOOL;
        pp_abr1 : BOOL; pp_abr2 : BOOL; pp_abr3 : BOOL; pp_abr4 : BOOL; pp_abr5 : BOOL; pp_abr6 : BOOL; pp_abr7 : BOOL; pp_abr8 : BOOL;
        pp_err : ARRAY[1..8] OF BOOL;
        pp_err1 : BOOL; pp_err2 : BOOL; pp_err3 : BOOL; pp_err4 : BOOL; pp_err5 : BOOL; pp_err6 : BOOL; pp_err7 : BOOL; pp_err8 : BOOL;
        pp_eid : ARRAY[1..8] OF UINT;
        pp_eid1 : UINT; pp_eid2 : UINT; pp_eid3 : UINT; pp_eid4 : UINT; pp_eid5 : UINT; pp_eid6 : UINT; pp_eid7 : UINT; pp_eid8 : UINT;
    MC_PP_Blend1 : MC_PositionProfile;
    MC_PP_Blend2 : MC_PositionProfile;
    MC_PP_Blend3 : MC_PositionProfile;
    MC_PP_Blend4 : MC_PositionProfile;
    MC_PP_Blend5 : MC_PositionProfile;
    MC_PP_Blend6 : MC_PositionProfile;
    MC_PP_Blend7 : MC_PositionProfile;
    MC_PP_Blend8 : MC_PositionProfile;
        pp_blend_exe : ARRAY[1..8] OF BOOL;
        pp_blend_done : BOOL;
        pp_blend_don : ARRAY[1..8] OF BOOL;
        pp_blend_don1 : BOOL; pp_blend_don2 : BOOL; pp_blend_don3 : BOOL; pp_blend_don4 : BOOL; pp_blend_don5 : BOOL; pp_blend_don6 : BOOL; pp_blend_don7 : BOOL; pp_blend_don8 : BOOL;
        pp_blend_busy : BOOL;
        pp_blend_bus : ARRAY[1..8] OF BOOL;
        pp_blend_bus1 : BOOL; pp_blend_bus2 : BOOL; pp_blend_bus3 : BOOL; pp_blend_bus4 : BOOL; pp_blend_bus5 : BOOL; pp_blend_bus6 : BOOL; pp_blend_bus7 : BOOL; pp_blend_bus8 : BOOL;
        pp_blend_active : BOOL;
        pp_blend_active_last : BOOL;
        pp_blend_act : ARRAY[1..8] OF BOOL;
        pp_blend_act1 : BOOL; pp_blend_act2 : BOOL; pp_blend_act3 : BOOL; pp_blend_act4 : BOOL; pp_blend_act5 : BOOL; pp_blend_act6 : BOOL; pp_blend_act7 : BOOL; pp_blend_act8 : BOOL;
        pp_blend_abr : ARRAY[1..8] OF BOOL;
        pp_blend_abr1 : BOOL; pp_blend_abr2 : BOOL; pp_blend_abr3 : BOOL; pp_blend_abr4 : BOOL; pp_blend_abr5 : BOOL; pp_blend_abr6 : BOOL; pp_blend_abr7 : BOOL; pp_blend_abr8 : BOOL;
        pp_blend_err : ARRAY[1..8] OF BOOL;
        pp_blend_err1 : BOOL; pp_blend_err2 : BOOL; pp_blend_err3 : BOOL; pp_blend_err4 : BOOL; pp_blend_err5 : BOOL; pp_blend_err6 : BOOL; pp_blend_err7 : BOOL; pp_blend_err8 : BOOL;
        pp_blend_eid : ARRAY[1..8] OF UINT;
        pp_blend_eid1 : UINT; pp_blend_eid2 : UINT; pp_blend_eid3 : UINT; pp_blend_eid4 : UINT; pp_blend_eid5 : UINT; pp_blend_eid6 : UINT; pp_blend_eid7 : UINT; pp_blend_eid8 : UINT;
    FB_S_Type_Motion_Par_Calc_3_Segments1 : FB_S_Type_Motion_Par_Calc_3_Segments;
    FB_TransitionCurveCalc1 : FB_TransitionCurveCalc;
        svel : REAL;
        evel : REAL;

    Axis_now_pulse : ARRAY[1..8] OF REAL;
    Axis_now_angel : ARRAY[1..8] OF REAL;
    Axis_last_angel : ARRAY[1..8] OF REAL;

    AxesGroup_State : USINT;        // 4(* AG_GS *)   轴组状态，定义见Group_State(GS_)
    AxesGroup_Axis_Sum_In_Group : USINT;    // 5(* AG_ASUM *) 轴组包含的实际的单轴数量
    AxesGroup_Axis_Num_In_Group : ARRAY[0..7] OF SINT;        // 6(* AG_ANUM *) 轴组里保存的轴号 ***需要初始化为-1***
    AxesGroup_Kin_Transform : USINT;        // 14(* AG_KT *)  该轴组绑定的运动学模型 0表示未绑定 1用来修改参数 2，3...表示运动学模型编号
    AxesGroup_DenHartA : ARRAY[1..9] OF REAL;         // 16(* AG_DHA *) 轴组绑定的运动学模型的DH参数
    AxesGroup_DenHartD : ARRAY[1..9] OF REAL;         // 52(* AG_DHD *) 轴组绑定的运动学模型的DH参数
    AxesGroup_JointDirections : ARRAY[1..8] OF SINT;  // 88(* AG_JD *)  每个轴的电机旋转方向
    AxesGroup_JointOffsets : ARRAY[1..8] OF DINT;     // 96(* AG_JO *)  每个轴的电子零点和机械零点的偏移量
    AxesGroup_JointGearRatios : ARRAY[1..8] OF REAL;          // 192(* AG_JGR *) U32电机分辨率/U32轴分辨率
    AxesGroup_Buffered_Transition_Parameter : REAL;           // 224(* AG_BTP *) 过渡参数,对于TM1/TM2，参数为速度比例；对于TM3/TM4，参数为距离
    AxesGroup_Interrupt_Deceleration : REAL;                  // 228(* AG_ID *) Interrupt减速度
    AxesGroup_Interrupt_Jerk : REAL;                          // 232(* AG_IJ *) Interrupt加加速度
    AxesGroup_Buffered_TCP_Velocity : REAL;                   // 236(* AG_BTV *) Buffer区中运动功能块的目标速度
    AxesGroup_Now_Axis_Motion_AimPulse : ARRAY[1..8] OF REAL; // 284(* AG_NMA *) 每个轴当前运动的目标脉冲值
    AxesGroup_Now_TCP_Position : ARRAY[1..6] OF REAL;         // 316(* AG_NTP *) 当前轴组所处的位姿
    AxesGroup_Run_Number : USINT;      // 415(* AG_RN *) 运动编号 从1开始，0为复位状态 1--255 不需要初始化，在运动FB中被赋值，需要复位
    AxesGroup_Busy : USINT;            // 416(* AG_BZ *) 当前轴组忙碌标志
    AxesGroup_Buffer_Busy : USINT;     // 417(* AG_BBZ *)轴组的缓冲区是否繁忙
    AxesGroup_Buffered_Transition_Mode : USINT;               // 421(* AG_BTM *) 过渡模式
    AxesGroup_Interrupt : USINT;       // 422(* AG_TINR *) 轴组运动打断标志
    AxesGroup_Now_TCP_Velocity : REAL;                        // 648(* AG_NTV *) 当前轴组的速度 单位：每周期多少mm TCP_V 都是非负数 初始值为0
    AxesGroup_Now_TCP_Direction : ARRAY[1..3] OF REAL;        // 652(* AG_NTD *) 当前轴组的运动方向 矢量值
    AxesGroup_TCP_Motion_Aimpos: ARRAY[1..6] OF REAL;         // 668(* AG_MAP *) 当前轴组运动的目标位姿
    AxesGroup_Axis_Buffer_AimPulse : ARRAY[1..8] OF REAL; // 724(* AG_BFAP *) 下一功能块目标TCP位置
    AxesGroup_JointLimits : ARRAY[1..16] OF REAL;             // 756(* AG_JL *)  轴组绑定的运动学模型的每个轴的角度限制

    axesgroup : USINT;
    vel : REAL;
    acc : REAL;
    dec : REAL;
    jerk : REAL;
    bufferMode : USINT;
    transitionMode : USINT;
    transitionParameter : REAL;
    pathchoice : USINT;
    coordSystem : USINT;
    circMode : USINT;
    endpoint : ARRAY[1..6] OF REAL;
    auxpoint : ARRAY[1..6] OF REAL;
    endpoint_tcp : ARRAY[1..6] OF REAL;
    auxpoint_tcp : ARRAY[1..6] OF REAL;

    tendpoint_index : USINT;
    tendpoint_tcp : ARRAY[1..6] OF REAL;
    tendpoint_dir : ARRAY[1..6] OF REAL;

    pointnumber : USINT;
    pointbuf_tcp : ARRAY[1..600] OF REAL;
    pointbuf_acp : ARRAY[1..800] OF REAL; 
    pointbuf_dtime : ARRAY[1..100] OF REAL;  

    BufCal_vs : REAL;
    BufCal_vel : REAL;
    BufCal_acc : REAL;
    BufCal_dec : REAL;
    BufCal_jerk : REAL;  

    blendVel : REAL;
    vmax : REAL;

    s1 : REAL; s2 : REAL; s3 : REAL; s4 : REAL; s5 : REAL; s6 : REAL; s7 : REAL;
    t1 : REAL; t2 : REAL; t3 : REAL; t4 : REAL; t5 : REAL; t6 : REAL; t7 : REAL;
    stage_v1 : REAL;
    stage_v2 : REAL;
    stage_v3 : REAL;
    curve_time : REAL;

    motionStop : BOOL;
    stop_time : REAL;
    stop_s : REAL;

    radius : REAL;    // 半径
    d : REAL;         // 弦长

    a : REAL;
    b : REAL;
    c : REAL;

    now_s : REAL;
    trans_s : REAL; // 过渡距离
    s_all : REAL;
    s_to_end : REAL;

    lastExe : BOOL;
    lastInt : BOOL;
    stage : USINT;
    group_RunNumberF : USINT;
    i : USINT;    // 循环变量
    j : USINT;    // 循环变量
    WRParRes : BOOL;
    now_motion_done : BOOL;
END_VAR

IF stage = 0(* FBS_WaitExec *) AND NOT Execute AND NOT Active THEN
    Busy := FALSE;
    Active := FALSE;
    Done := FALSE;
    Error := FALSE;
    ErrorID := 0(* ER_NoError *);
    CommandAborted := FALSE;

    pp_done := FALSE;
    pp_blend_done := FALSE;
    FOR i := 1 TO 8 DO
        pp_exe[i] := FALSE;
        pp_blend_exe[i] := FALSE;
    END_FOR;
    group_RunNumberF := 0;
    pp_active_last := FALSE;
END_IF;

IF stage = 0 AND Execute AND NOT lastExe AND NOT Active THEN
    IF AxesGroup < 0 OR AxesGroup > 20 THEN
        ErrorID := 16#05;
    END_IF;
    AxesGroup_State := MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 4(* AG_GS *));
    IF AxesGroup_State = 0(* GS_GroupDisabled *) OR AxesGroup_State = 1(* GS_GroupErrorstop *) THEN
        ErrorID := 16#01 (* ER_StateError *);
    ELSIF AxesGroup_State = 4(* GS_GroupStopping *) THEN
        ErrorID := 16#20(* ER_AxisState_Stopping_Error *);
    END_IF;
    IF Velocity < 0 OR Acceleration <= 0 OR Deceleration <= 0 OR Jerk < 0 THEN
        ErrorID := 16#02(* ER_ParameterError *);
    END_IF;
    IF CoordSystem <> 1(* CS_ACS *) AND CoordSystem <> 2(* CS_MCS *) AND CoordSystem <> 3(* CS_PCS *) THEN
        ErrorID := 16#06(* ER_CoordSystem_ParameterError *);
    END_IF;
    IF BufferMode >= 6 THEN
        ErrorID := 16#02(* ER_ParameterError *);
    END_IF;
    IF TransitionMode >= 5 THEN
        ErrorID := 16#02(* ER_ParameterError *);
    END_IF;

    IF ErrorID <> 0(* ER_NoError *) THEN
        Error := TRUE;
    ELSE
        AxesGroup_Axis_Sum_In_Group := MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 5(* AG_ASUM *));
        AxesGroup_Kin_Transform := MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 14(* AG_KT *));
        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            AxesGroup_JointLimits[i * 2 - 1] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 756(* AG_JL *) + (i - 1) * 2 * 4);
            AxesGroup_JointLimits[i * 2] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 756(* AG_JL *) + (i * 2 - 1) * 4);
            AxesGroup_DenHartA[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 16(* AG_DHA *) + (i - 1) * 4);
            AxesGroup_DenHartD[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 52(* AG_DHD *) + (i - 1) * 4);
            AxesGroup_JointDirections[i] := USINT_TO_SINT(MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 88(* AG_JD *) + (i - 1)));
            AxesGroup_JointOffsets[i] := MC_RdAxesPar_D(AxesNum := AxesGroup,AddrOffset := 96(* AG_JO *) + (i - 1) * 4);
            AxesGroup_JointGearRatios[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 192(* AG_JGR *) + (i - 1) * 4);

            AxesGroup_Axis_Num_In_Group[i - 1] := USINT_TO_SINT(MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 6(* AG_ANUM *) + i - 1));
        END_FOR;
        AxesGroup_DenHartA[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 16(* AG_DHA *) + (i - 1) * 4);
        AxesGroup_DenHartD[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 52(* AG_DHD *) + (i - 1) * 4);

        AxesGroupOut := AxesGroup;
        axesgroup := AxesGroup;
        vel := Velocity;
        acc := Acceleration;
        dec := Deceleration;
        jerk := Jerk;
        bufferMode := BufferMode;
        transitionMode := TransitionMode;
        transitionParameter := TransitionParameter;
        circMode := CircMode;
        pathchoice := PathChoice;
        coordSystem := CoordSystem;
        FOR i := 1 TO 6 DO
            endpoint[i] := EndPoint[i];
            auxpoint[i] := AuxPoint[i];
        END_FOR;

        AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
        group_RunNumberF := AxesGroup_Run_Number;

        Busy := TRUE;
        stage := 1(* FBS_WaitBuffer *);
    END_IF;
END_IF;

IF stage = 1(* FBS_WaitBuffer *) OR stage = 2(* FBS_InBuffer *) THEN
    AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
    IF AxesGroup_Run_Number <> group_RunNumberF THEN
        CommandAborted := TRUE;
        Busy := FALSE;
        stage := 0(* FBS_WaitExec *);
    END_IF;
END_IF;

IF stage = 1(* FBS_WaitBuffer *) THEN
    IF bufferMode = 0(* BM_Aborting *) THEN
        AxesGroup_Buffered_Transition_Mode := transitionMode;
        CASE AxesGroup_Buffered_Transition_Mode OF
            0(* TM_None *) :  ;
            1(* TM_StartV *) ,
            2(* TM_ConstantV *) : AxesGroup_Buffered_Transition_Parameter := transitionParameter;
            3(* TM_CornerDis *),
            4(* TM_MaxCornerDeviation *) : ErrorID := 16#27(* ER_Buf_Trans_Combination_NotSupported *);
        END_CASE;

        IF ErrorID = 0 THEN
            group_RunNumberF := AxesGroup_Run_Number MOD 255 + 1; // [1,255]
            WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *),Data := group_RunNumberF);
        END_IF;
    ELSE
        AxesGroup_Buffer_Busy := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *));
        IF AxesGroup_Buffer_Busy = 0 THEN
            AxesGroup_Buffered_Transition_Mode := transitionMode;
            CASE bufferMode OF
                1(* BM_Buffered *) :
                    CASE AxesGroup_Buffered_Transition_Mode OF
                        0(* TM_None *) :  ;
                        1(* TM_StartV *) ,
                        2(* TM_ConstantV *) ,
                        3(* TM_CornerDis *),
                        4(* TM_MaxCornerDeviation *) : ErrorID := 16#27(* ER_Buf_Trans_Combination_NotSupported *);
                    END_CASE;
                2(* BM_Blendinglow *) ,
                3(* BM_BlendingPrevious *) ,
                4(* BM_BlendingNext *) ,
                5(* BM_BlendingHigh *) :
                    CASE AxesGroup_Buffered_Transition_Mode OF
                        0(* TM_None *) : ErrorID := 16#27(* ER_Buf_Trans_Combination_NotSupported *);
                        1(* TM_StartV *) ,
                        2(* TM_ConstantV *) ,
                        3(* TM_CornerDis *) ,
                        4(* TM_MaxCornerDeviation *) : AxesGroup_Buffered_Transition_Parameter := transitionParameter;
                    END_CASE;
            END_CASE;
        END_IF;
    END_IF;

    IF ErrorID = 0(* ER_NoError *) THEN
        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *),Data := bufferMode);
        WRParRes := MC_WrAxesPar_FL(AxesNum := axesgroup,AddrOffset := 236(* AG_BTV *), Data := vel);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 421(* AG_BTM *),Data := AxesGroup_Buffered_Transition_Mode);
        WRParRes := MC_WrAxesPar_FL(AxesNum := axesgroup,AddrOffset := 224(* AG_BTP *),Data := AxesGroup_Buffered_Transition_Parameter);

        stage := 2(* FBS_InBuffer *);
    ELSE
        Error := TRUE;
        stage := 0(* FBS_WaitExec *);
    END_IF;
END_IF;

IF stage = 2(* FBS_InBuffer *) THEN
    AxesGroup_Busy := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *));
    IF AxesGroup_Busy = 0 THEN
        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *),Data := 0); 
        AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));

        IF transitionMode = 0 AND AxesGroup_Now_TCP_Velocity > 1 THEN
            Error := TRUE;
            ErrorID := 16#0B(* ER_Transition_Compute_Error *);
            stage := 0(* FBS_WaitExec *);
        ELSE
            stage := 3(* FBS_CalcRoute *);
        END_IF;
    END_IF;
END_IF;

IF stage = 3(* FBS_CalcRoute *) THEN
    IF bufferMode = 0 (* BM_Aborting *) THEN
        FOR i := 1 TO 6 DO
            nowpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
            strpoint_tcp[i] := nowpoint_tcp[i];
        END_FOR;

        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            Axis_now_pulse[i] := MC_RdAxisPar_FL(AxisNum := SINT_TO_USINT(AxesGroup_Axis_Num_In_Group[i - 1]),AddrOffset := 76);
            Axis_now_angel[i] := (Axis_now_pulse[i] - AxesGroup_JointOffsets[i]) * AxesGroup_JointDirections[i] / AxesGroup_JointGearRatios[i];
            Axis_last_angel[i] := Axis_now_angel[i];
        END_FOR;
    ELSE
        FOR i := 1 TO 6 DO
            nowpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
            strpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 668(* AG_MAP *) + 4 * (i - 1));
        END_FOR;

        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            AxesGroup_Now_Axis_Motion_AimPulse[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 284(* AG_NMA *) + (i - 1) * 4);
            Axis_last_angel[i] := (AxesGroup_Now_Axis_Motion_AimPulse[i] - AxesGroup_JointOffsets[i]) * AxesGroup_JointDirections[i] / AxesGroup_JointGearRatios[i];
        END_FOR;
    END_IF;

    IF coordSystem = 1(* CS_ACS *) THEN
        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            endpoint_tcp[i] := endpoint[i] + Axis_now_angel[i];
            auxpoint_tcp[i] := auxpoint[i] + Axis_now_angel[i];
        END_FOR;
        FB_ACS_MCS_KinTransform1(axesgroup_kin := AxesGroup_Kin_Transform,ACS_JointAngle := endpoint_tcp[1],DenHartA := AxesGroup_DenHartA[1],DenHartD := AxesGroup_DenHartD[1],MCS_X => endpoint_tcp[1],MCS_Y => endpoint_tcp[2],MCS_Z => endpoint_tcp[3],MCS_RX => endpoint_tcp[4],MCS_RY => endpoint_tcp[5],MCS_RZ => endpoint_tcp[6]); 
        FB_ACS_MCS_KinTransform1(axesgroup_kin := AxesGroup_Kin_Transform,ACS_JointAngle := auxpoint_tcp[1],DenHartA := AxesGroup_DenHartA[1],DenHartD := AxesGroup_DenHartD[1],MCS_X => auxpoint_tcp[1],MCS_Y => auxpoint_tcp[2],MCS_Z => auxpoint_tcp[3],MCS_RX => auxpoint_tcp[4],MCS_RY => auxpoint_tcp[5],MCS_RZ => auxpoint_tcp[6]); 
    ELSIF  coordSystem = 2(* CS_MCS *) THEN
        FOR i := 1 TO 6 DO
            endpoint_tcp[i] := endpoint[i] + strpoint_tcp[i];
            auxpoint_tcp[i] := auxpoint[i] + strpoint_tcp[i];
        END_FOR;
    END_IF;

    IF debug_print_PATH THEN
        debug_printl_real1(0);
        debug_printt_real1(nowpoint_tcp[1]); debug_printt_real1(nowpoint_tcp[2]); debug_printt_real1(nowpoint_tcp[3]); debug_printt_real1(nowpoint_tcp[4]); debug_printt_real1(nowpoint_tcp[5]); debug_printl_real1(nowpoint_tcp[6]);
        debug_printt_real1(auxpoint_tcp[1]); debug_printt_real1(auxpoint_tcp[2]); debug_printt_real1(auxpoint_tcp[3]); debug_printt_real1(auxpoint_tcp[4]); debug_printt_real1(auxpoint_tcp[5]); debug_printl_real1(auxpoint_tcp[6]);
        debug_printt_real1(endpoint_tcp[1]); debug_printt_real1(endpoint_tcp[2]); debug_printt_real1(endpoint_tcp[3]); debug_printt_real1(endpoint_tcp[4]); debug_printt_real1(endpoint_tcp[5]); debug_printl_real1(endpoint_tcp[6]);
    END_IF;

    FOR i := 1 TO 6 DO
        circle_path[i] := strpoint_tcp[i];
    END_FOR;

    FB_CircularPointsCal1(CircMode := circMode,PathChoice := pathchoice,
                          NowPosition := strpoint_tcp[1],AuxPosition := auxpoint_tcp[1],EndPosition := endpoint_tcp[1],PointNum := 99,
                          Error => cirerror,ErrorID => cirerrrorid,PointBuf => circle_path[7], CenterPoint => center_point[1]);

    IF circMode = 1(* CM_Border *) THEN
        circMode := 2(* CM_Center *);
        auxpoint_tcp[1] := center_point[1];
        auxpoint_tcp[2] := center_point[2];
        auxpoint_tcp[3] := center_point[3];
    END_IF;

    radius := ESQR((circle_path[1] - center_point[1]) * (circle_path[1] - center_point[1]) +
                   (circle_path[2] - center_point[2]) * (circle_path[2] - center_point[2]) +
                   (circle_path[3] - center_point[3]) * (circle_path[3] - center_point[3]));

    path_s[1] := 0;
    FOR i := 2 TO 100 DO
        d := ESQR((circle_path[6 * (i - 1) + 1] - circle_path[6 * (i - 2) + 1]) * (circle_path[6 * (i - 1) + 1] - circle_path[6 * (i - 2) + 1]) +
                  (circle_path[6 * (i - 1) + 2] - circle_path[6 * (i - 2) + 2]) * (circle_path[6 * (i - 1) + 2] - circle_path[6 * (i - 2) + 2]) +
                  (circle_path[6 * (i - 1) + 3] - circle_path[6 * (i - 2) + 3]) * (circle_path[6 * (i - 1) + 3] - circle_path[6 * (i - 2) + 3]));
        path_s[i] := 2.0 * radius * EASIN(d / radius / 2) + path_s[i - 1];
    END_FOR;

    IF debug_print_PATH THEN
        debug_printl_real1(0);
        FOR i := 1 TO 100 DO
            FOR j := 1 TO 6 DO
                debug_printt_real1(circle_path[6 * (i - 1) + j]);
            END_FOR;
                debug_printl_real1(path_s[i]);
        END_FOR;
    END_IF;

    pointbuf_dtime[1] := 0;    // 将pointbuf清空

    AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));

    IF transitionMode <> 0 AND AxesGroup_Now_TCP_Velocity <> 0 THEN
        FOR i := 1 TO 3 DO
            AxesGroup_Now_TCP_Direction[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 652(* AG_NTD *) + 4 * (i - 1));
        END_FOR;

        IF bufferMode > 1 THEN
            FOR i := 1 TO 6 DO
                strpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 668(* AG_MAP *) + 4 * (i - 1));
            END_FOR;
        END_IF;

        CASE transitionMode OF
            1(* TM_StartV *) :
                IF jerk = 0 THEN
                    curve_time := vel / acc;
                    trans_s := Motion_Delta_S(vs := 0,acc := acc,dec := 0,jerk := 0,d_t := curve_time);
                ELSE
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := 0,vt := vel,acc := acc,dec := 0,jerk := jerk,t1 => t1,t2 => t2,t3 => t3,s1 => s1,s2 => s2,s3 => s3,t_3seg => curve_time,s_3seg => trans_s);
                END_IF;

                FOR i := 1 TO 100 DO
                    IF path_s[i] >= trans_s THEN
                        tendpoint_index := i;
                        EXIT;
                    END_IF;
                END_FOR;
                curve_time := curve_time + (path_s[tendpoint_index] - trans_s) / vel;

                svel := AxesGroup_Now_TCP_Velocity * 1000; evel := vel;
            2(* TM_ConstantV *) :
                IF jerk = 0 THEN
                    curve_time := (vel * transitionParameter) / acc;
                    trans_s := Motion_Delta_S(vs := 0,acc := acc,dec := 0,jerk := 0,d_t := curve_time);
                ELSE
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := 0,vt := vel,acc := acc,dec := 0,jerk := jerk,t1 =&gt; t1,t2 =&gt; t2,t3 =&gt; t3,s1 =&gt; s1,s2 =&gt; s2,s3 =&gt; s3,t_3seg =&gt; ,s_3seg =&gt; );
                    stage_v1 := 0;
                    stage_v2 := 0.5 * jerk * t1 * t1;
                    stage_v3 := stage_v2 + jerk * t1 * t2;
                    IF 0.5 * jerk * t1 * t1 < vel * transitionParameter THEN
                        curve_time := ESQR(vel * transitionParameter * 2 / jerk);
                        trans_s := Motion_Delta_S(vs := 0,acc := 0,dec := 0,jerk := jerk,d_t := ESQR(vel * transitionParameter * 2 / jerk));
                    ELSIF 0.5 * jerk * t1 * t1 + acc * t2 < vel * transitionParameter THEN
                        trans_s := s1 + Motion_Delta_S(vs := stage_v2,acc := acc,dec := 0,jerk := 0,d_t := (vel * transitionParameter - 0.5 * jerk * t1 * t1) / acc);
                        curve_time := (vel * transitionParameter - 0.5 * jerk * t1 * t1) / acc + t1;
                    ELSE
                        trans_s := s1 + s2 + Motion_Delta_S(vs := stage_v3,acc := acc,dec := 0,jerk := -jerk,d_t := (vel * transitionParameter - 0.5 * jerk * t1 * t1) / acc);
                        curve_time := (vel * transitionParameter - 0.5 * jerk * t1 * t1) / acc + t1 + t2;
                    END_IF;
                END_IF;

                FOR i := 1 TO 100 DO
                    IF path_s[i] >= trans_s THEN
                        tendpoint_index := i;
                        EXIT;
                    END_IF;
                END_FOR;
                curve_time := curve_time + (path_s[tendpoint_index] - trans_s) / (vel * transitionParameter);

                svel := AxesGroup_Now_TCP_Velocity * 1000; evel := svel;
            3(* TM_CornerDis *),
            4(* TM_MaxCornerDeviation *) :
                FOR i := 1 TO 100 DO
                    IF path_s[i] >= transitionParameter THEN
                        tendpoint_index := i;
                        EXIT;
                    END_IF;
                END_FOR;

                trans_s := path_s[i];
                IF jerk = 0 THEN
                    t1 := vel / acc;    // 开始速度为0
                    t3 := vel / dec;    // 结束速度为0
                    s1 := Motion_Delta_S(vs := 0,acc := acc,dec := 0,jerk := 0,d_t := t1);
                    s3 := Motion_Delta_S(vs := vel,acc := 0,dec := dec,jerk := 0,d_t := t3);
                    s2 := path_s[100] - s1 - s3;
                    t2 := s2 / vel;
                    vmax := vel;
                    IF t2 < 0 THEN
                        vmax := ESQR((2 * acc * dec * path_s[100]) / (acc + dec));
                        t1 := vmax / acc;    // 开始速度为0
                        t2 := 0;
                        t3 := vmax / dec;    // 结束速度为0
                        s1 := acc * t1 * t1 / 2;
                        s2 := 0;
                        s3 := path_s[100] - s1;
                    END_IF;

                    IF trans_s <= s1 THEN
                        curve_time := ESQR(2 * trans_s / acc);
                        evel := acc * curve_time;
                    ELSIF trans_s <= s1 + s2 THEN
                        curve_time := t1 + (trans_s - s1) / vmax;
                        evel := vmax;
                    ELSE
                        a := dec / 2.0;
                        b := -vmax;
                        c := trans_s - s1 - s2;
                        curve_time := (ESQR(b * b - 4 * a * c) - b) / (2 * a);
                        IF curve_time < 0 OR curve_time > t3 THEN
                            curve_time := ( -b - ESQR(b * b - 4 * a * c)) / (2 * a);
                        END_IF;
                        curve_time := t1 + t2 + curve_time;
                        evel := vmax - dec * curve_time;
                    END_IF;
                ELSE
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := 0,vt := vel,acc := acc,dec := 0,jerk := jerk,t1 =&gt; t1,t2 =&gt; t2,t3 =&gt; t3,s1 =&gt; s1,s2 =&gt; s2,s3 =&gt; s3,t_3seg =&gt; ,s_3seg =&gt; );
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := vel,vt := 0,acc := 0,dec := dec,jerk := jerk,t1 =&gt; t5,t2 =&gt; t6,t3 =&gt; t7,s1 =&gt; s5,s2 =&gt; s6,s3 =&gt; s7,t_3seg =&gt; ,s_3seg =&gt; );
                    s4 := path_s[100] - s1 - s2 - s3 - s5 - s6 - s7;
                    t4 := s4 / vel;
                    IF t4 > 0 THEN
                        stage_v1 := 0;
                        stage_v2 := 0.5 * jerk * t1 * t1;
                        stage_v3 := stage_v2 + jerk * t1 * t2;
                        IF trans_s < s1 THEN
                            curve_time := Solve_Cubic_Equation(a := jerk / 6.0,b := 0,c := 0,d := -trans_s, LimitL := 0, LimitR := t1);
                            evel := stage_v1 + 0.5 * jerk * curve_time * curve_time;
                        ELSIF trans_s < s1 + s2 THEN
                            trans_s := trans_s - s1;
                            IF - 0.001 < trans_s AND trans_s < 0.001 THEN trans_s := 0; END_IF;
                            a := acc / 2.0;
                            b := stage_v2;
                            c := -trans_s;
                            curve_time := (ESQR(b * b - 4 * a * c) - b) / (2 * a);
                            IF curve_time < 0 OR curve_time > t2 THEN
                                curve_time := ( -b - ESQR(b * b - 4 * a * c)) / (2 * a);
                            END_IF;
                            evel := stage_v2 + acc * curve_time;
                            curve_time := curve_time + t1;
                        ELSIF trans_s < s1 + s2 + s3 THEN
                            trans_s := trans_s - s1 - s2;
                            curve_time := Solve_Cubic_Equation(a := jerk / - 6.0,b := jerk * t1 / 2,c := stage_v3,d := -trans_s, LimitL := 0, LimitR := t3);
                            evel := stage_v3 + jerk * t1 * curve_time - 0.5 * jerk * curve_time * curve_time;
                            curve_time := curve_time + t1 + t2;
                        ELSIF trans_s < s1 + s2 + s3 + s4 THEN
                            trans_s := trans_s - s1 - s2 - s3;
                            curve_time := trans_s / vel;
                            evel := vel;
                            curve_time := curve_time + t1 + t2 + t3;
                        END_IF;
                        trans_s := path_s[i];
                    END_IF;
                END_IF;
                svel := AxesGroup_Now_TCP_Velocity * 1000;
        END_CASE;

        FOR i := 1 TO 100 DO
            IF path_s[i] >= trans_s THEN
                tendpoint_index := i;
                EXIT;
            END_IF;
        END_FOR;

        FOR i := 1 TO 6 DO
            tendpoint_tcp[i] := circle_path[6 * (tendpoint_index - 1) + i];
            tendpoint_dir[i] := (circle_path[6 * (tendpoint_index - 1) + i] - circle_path[6 * (tendpoint_index - 2) + i]) / (path_s[tendpoint_index] - path_s[tendpoint_index - 1]);
        END_FOR;

        FB_TransitionCurveCalc1(spoint_TCP := nowpoint_tcp[1], svel := svel,svel_dir := AxesGroup_Now_TCP_Direction[1],
                                epoint_TCP := tendpoint_tcp[1],evel := evel,evel_dir := tendpoint_dir[1],pointnumber := 20,
                                PointBuf_TCP => pointbuf_tcp[1],PointBuf_dtime => pointbuf_dtime[1]);
        pointbuf_dtime[20 + 1] := curve_time; pointbuf_dtime[20 + 2] := 0;

        IF debug_print_PATH THEN
            debug_printl_real1(0);
            FOR i := 1 TO 20 DO
                FOR j := 1 TO 6 DO
                    debug_printt_real1(pointbuf_tcp[6 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_dtime[i]);
            END_FOR;
        END_IF;
    END_IF;

    BufCal_vel := vel;
    BufCal_acc := acc;
    BufCal_dec := dec;
    BufCal_jerk := jerk;
    path_point_num := 100;

    FB_CircularPointBufCal1(circle_path := circle_path[1], center_point := center_point[1],path_s := path_s[1],path_point_num := path_point_num,pathchoice := pathchoice,
                            vs := 0,vel := BufCal_vel,acc := BufCal_acc,dec := BufCal_dec,jerk := BufCal_jerk,
                            pointbuf_tcp => pointbuf_tcp[1],pointbuf_dtime => pointbuf_dtime[1],pointnumber => pointnumber);

    FOR i := 0 TO pointnumber - 1 DO
        FB_MCS_ACS_KinTransform1(axesgroup_kin := AxesGroup_Kin_Transform,TCP_TargetPos := ADR(pointbuf_tcp[6 * i + 1]),DenHartA := ADR(AxesGroup_DenHartA[1]),DenHartD := ADR(AxesGroup_DenHartD[1]),JointLimits := ADR(AxesGroup_JointLimits[1]),JointAngles_now := ADR(Axis_last_angel[1]),JointAngle1 => pointbuf_acp[8 * i + 1],JointAngle2 => pointbuf_acp[8 * i + 2],JointAngle3 => pointbuf_acp[8 * i + 3],JointAngle4 => pointbuf_acp[8 * i + 4],JointAngle5 => pointbuf_acp[8 * i + 5],JointAngle6 => pointbuf_acp[8 * i + 6],JointAngle7 => pointbuf_acp[8 * i + 7],JointAngle8 => pointbuf_acp[8 * i + 8],Error => Error,ErrorID => );

        FOR j := 1 TO AxesGroup_Axis_Sum_In_Group DO
            Axis_last_angel[j] := pointbuf_acp[8 * i + j];
        END_FOR;

        IF Error THEN
            ErrorID := 9000 + i;
            EXIT;
        END_IF;
    END_FOR;

    FOR i := 1 TO pointnumber DO
        pointbuf_pus1[i] := pointbuf_acp[8 * (i - 1) + 1] * AxesGroup_JointGearRatios[1] * AxesGroup_JointDirections[1] + AxesGroup_JointOffsets[1];
        pointbuf_pus2[i] := pointbuf_acp[8 * (i - 1) + 2] * AxesGroup_JointGearRatios[2] * AxesGroup_JointDirections[2] + AxesGroup_JointOffsets[2];
        pointbuf_pus3[i] := pointbuf_acp[8 * (i - 1) + 3] * AxesGroup_JointGearRatios[3] * AxesGroup_JointDirections[3] + AxesGroup_JointOffsets[3];
        pointbuf_pus4[i] := pointbuf_acp[8 * (i - 1) + 4] * AxesGroup_JointGearRatios[4] * AxesGroup_JointDirections[4] + AxesGroup_JointOffsets[4];
        pointbuf_pus5[i] := pointbuf_acp[8 * (i - 1) + 5] * AxesGroup_JointGearRatios[5] * AxesGroup_JointDirections[5] + AxesGroup_JointOffsets[5];
        pointbuf_pus6[i] := pointbuf_acp[8 * (i - 1) + 6] * AxesGroup_JointGearRatios[6] * AxesGroup_JointDirections[6] + AxesGroup_JointOffsets[6];
        pointbuf_pus7[i] := pointbuf_acp[8 * (i - 1) + 7] * AxesGroup_JointGearRatios[7] * AxesGroup_JointDirections[7] + AxesGroup_JointOffsets[7];
        pointbuf_pus8[i] := pointbuf_acp[8 * (i - 1) + 8] * AxesGroup_JointGearRatios[8] * AxesGroup_JointDirections[8] + AxesGroup_JointOffsets[8];
    END_FOR;

    IF pointnumber = 0 THEN
        Error := TRUE;
        ErrorID := 16#15(* ER_NoPathPoint *);

        debug_printl_real1(-UINT_TO_REAL(ErrorID));
        debug_printt_real1(nowpoint_tcp[1]); debug_printt_real1(nowpoint_tcp[2]); debug_printt_real1(nowpoint_tcp[3]); debug_printt_real1(nowpoint_tcp[4]); debug_printt_real1(nowpoint_tcp[5]); debug_printl_real1(nowpoint_tcp[6]);
        debug_printt_real1(auxpoint_tcp[1]); debug_printt_real1(auxpoint_tcp[2]); debug_printt_real1(auxpoint_tcp[3]); debug_printt_real1(auxpoint_tcp[4]); debug_printt_real1(auxpoint_tcp[5]); debug_printl_real1(auxpoint_tcp[6]);
        debug_printt_real1(endpoint_tcp[1]); debug_printt_real1(endpoint_tcp[2]); debug_printt_real1(endpoint_tcp[3]); debug_printt_real1(endpoint_tcp[4]); debug_printt_real1(endpoint_tcp[5]); debug_printl_real1(endpoint_tcp[6]);
    ELSE
        FOR i := 1 TO pointnumber DO
            IF pointbuf_dtime[i] <= 0.0001 THEN
                Error := TRUE;
                ErrorID := 16#28(* ER_Path_Dtime_Zero *);
                EXIT;
            END_IF;
        END_FOR;
    END_IF;

    IF NOT Error THEN
        stage := 4(* FBS_Moving *);

        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            pp_exe[i] := TRUE;
        END_FOR;

        IF debug_print_TCP THEN
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO 6 DO
                    debug_printt_real1(pointbuf_tcp[6 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_dtime[i]);
            END_FOR;
        END_IF;

        IF debug_print_ACP THEN
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO AxesGroup_Axis_Sum_In_Group - 1 DO
                    debug_printt_real1(pointbuf_acp[8 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_acp[8 * (i - 1) + j]);
            END_FOR;
        END_IF;
    ELSE
        stage := 0(* FBS_WaitExec *);
    END_IF;
END_IF;

MC_PositionProfile1(Axis_num := axesgroup + 0,Execute := pp_exe[1],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus1[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut =&gt; ,Done =&gt; pp_don1,Busy =&gt; pp_bus1,Active =&gt; pp_act1,CommandAborted =&gt; pp_abr1,Error =&gt; pp_err1,ErrorID =&gt; pp_eid1) ;
MC_PositionProfile2(Axis_num := axesgroup + 1,Execute := pp_exe[2],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus2[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don2,Busy => pp_bus2,Active => pp_act2,CommandAborted => pp_abr2,Error => pp_err2,ErrorID => pp_eid2) ;
MC_PositionProfile3(Axis_num := axesgroup + 2,Execute := pp_exe[3],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus3[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don3,Busy => pp_bus3,Active => pp_act3,CommandAborted => pp_abr3,Error => pp_err3,ErrorID => pp_eid3) ;
MC_PositionProfile4(Axis_num := axesgroup + 3,Execute := pp_exe[4],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus4[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don4,Busy => pp_bus4,Active => pp_act4,CommandAborted => pp_abr4,Error => pp_err4,ErrorID => pp_eid4) ;
MC_PositionProfile5(Axis_num := axesgroup + 4,Execute := pp_exe[5],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus5[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don5,Busy => pp_bus5,Active => pp_act5,CommandAborted => pp_abr5,Error => pp_err5,ErrorID => pp_eid5) ;
MC_PositionProfile6(Axis_num := axesgroup + 5,Execute := pp_exe[6],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus6[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don6,Busy => pp_bus6,Active => pp_act6,CommandAborted => pp_abr6,Error => pp_err6,ErrorID => pp_eid6) ;
MC_PositionProfile7(Axis_num := axesgroup + 6,Execute := pp_exe[7],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus7[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don7,Busy => pp_bus7,Active => pp_act7,CommandAborted => pp_abr7,Error => pp_err7,ErrorID => pp_eid7) ;
MC_PositionProfile8(Axis_num := axesgroup + 7,Execute := pp_exe[8],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus8[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don8,Busy => pp_bus8,Active => pp_act8,CommandAborted => pp_abr8,Error => pp_err8,ErrorID => pp_eid8) ;

pp_don[1] := pp_don1; pp_don[2] := pp_don2; pp_don[3] := pp_don3; pp_don[4] := pp_don4; pp_don[5] := pp_don5; pp_don[6] := pp_don6; pp_don[7] := pp_don7; pp_don[8] := pp_don8;
pp_bus[1] := pp_bus1; pp_bus[2] := pp_bus2; pp_bus[3] := pp_bus3; pp_bus[4] := pp_bus4; pp_bus[5] := pp_bus5; pp_bus[6] := pp_bus6; pp_bus[7] := pp_bus7; pp_bus[8] := pp_bus8;
pp_act[1] := pp_act1; pp_act[2] := pp_act2; pp_act[3] := pp_act3; pp_act[4] := pp_act4; pp_act[5] := pp_act5; pp_act[6] := pp_act6; pp_act[7] := pp_act7; pp_act[8] := pp_act8;
pp_abr[1] := pp_abr1; pp_abr[2] := pp_abr2; pp_abr[3] := pp_abr3; pp_abr[4] := pp_abr4; pp_abr[5] := pp_abr5; pp_abr[6] := pp_abr6; pp_abr[7] := pp_abr7; pp_abr[8] := pp_abr8;

pp_done := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_done := pp_done AND pp_don[i];
END_FOR;
pp_busy := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_busy := pp_busy AND pp_bus[i];
END_FOR;
pp_active := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_active := pp_active AND pp_act[i];
END_FOR;

IF stage = 4(* FBS_Moving *) THEN
    ;
END_IF;

IF stage = 4(* FBS_Moving *) THEN
    Active := TRUE;

    AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
    IF AxesGroup_Run_Number <> group_RunNumberF AND group_RunNumberF <> 0 THEN
        CommandAborted := TRUE;

        AxesGroup_Buffered_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 236(* AG_BTV *));
        AxesGroup_Buffered_Transition_Mode := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 421(* AG_BTM *));
        AxesGroup_Buffered_Transition_Parameter := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 224(* AG_BTP *));

        CASE AxesGroup_Buffered_Transition_Mode OF
            0(* TM_None *) : ;
            1(* TM_StartV *) : blendVel  := vel;
            2(* TM_ConstantV *) : blendVel := AxesGroup_Buffered_TCP_Velocity;
        END_CASE;

        stage := 5(* FBS_CalcAbort *);
    END_IF;

    AxesGroup_Interrupt := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 422(* AG_TINR *));
    IF AxesGroup_Interrupt = 1 THEN
        AxesGroup_Interrupt_Deceleration := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 228(* AG_ID *));
        AxesGroup_Interrupt_Jerk := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 232(* AG_IJ *));

        stage := 5(* FBS_CalcAbort *);
    END_IF;

    AxesGroup_Buffer_Busy := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *));
    IF AxesGroup_Buffer_Busy > 1 THEN
        AxesGroup_Buffered_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 236(* AG_BTV *));

        CASE AxesGroup_Buffer_Busy OF
            2(* BM_Blendinglow *) : IF vel < AxesGroup_Buffered_TCP_Velocity THEN blendVel := vel; ELSE blendVel := AxesGroup_Buffered_TCP_Velocity; END_IF;
            3(* BM_BlendingPrevious *) : blendVel := vel;
            4(* BM_BlendingNext *) : blendVel := AxesGroup_Buffered_TCP_Velocity;
            5(* BM_BlendingHigh *) : IF vel < AxesGroup_Buffered_TCP_Velocity THEN blendVel := AxesGroup_Buffered_TCP_Velocity; ELSE blendVel := vel; END_IF;
        END_CASE;

        AxesGroup_Buffered_Transition_Mode := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 421(* AG_BTM *));
        AxesGroup_Buffered_Transition_Parameter := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 224(* AG_BTP *));

        CASE AxesGroup_Buffered_Transition_Mode OF
            1(* TM_StartV *) ,  // 不重新规划轨迹，正常减速到过渡速度
            2(* TM_ConstantV *) : stage := 7(* FBS_RunBlend *); // 过渡速度由BufferMode和TransitionParameter共同决定，是否需要重新规划轨迹速度待考虑
            3(* TM_CornerDis *),
            4(* TM_MaxCornerDeviation *) : stage := 7(* FBS_CalcBlend *); // 先不重新规划速度曲线
        END_CASE;
    END_IF;

    IF pp_busy AND pp_active AND NOT pp_active_last THEN
        AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
        group_RunNumberF := AxesGroup_Run_Number MOD 255 + 1; // [1,255]
        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *),Data := group_RunNumberF);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *),Data := 0); 
        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *), Data := 1);
        AxesGroup_Now_Axis_Motion_AimPulse[1] := pointbuf_pus1[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[2] := pointbuf_pus2[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[3] := pointbuf_pus3[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[4] := pointbuf_pus4[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[5] := pointbuf_pus5[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[6] := pointbuf_pus6[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[7] := pointbuf_pus7[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[8] := pointbuf_pus8[pointnumber];
        FOR i := 1 TO 6 DO
            AxesGroup_TCP_Motion_Aimpos[i] := pointbuf_tcp[REAL_TO_UDINT(6 * (pointnumber - 1) + i)];
            WRParRes := MC_WrAxesPar_FL(AxesNum := axesgroup,AddrOffset := 668(* AG_MAP *) + (i - 1) * 4, Data := AxesGroup_TCP_Motion_Aimpos[i]);
            WRParRes := MC_WrAxesPar_FL(AxesNum := axesgroup,AddrOffset := 284(* AG_NMA *) + (i - 1) * 4, Data := AxesGroup_Now_Axis_Motion_AimPulse[i]);
        END_FOR;
    END_IF;
    pp_active_last := pp_active;

    IF pp_done THEN
        Done := TRUE;
        Error := FALSE;
        Busy := FALSE;
        Active := FALSE;
        stage := 0(* FBS_WaitExec *);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *), Data := 0);
    END_IF;
END_IF;

IF stage = 5(* FBS_CalcAbort *) THEN
    FOR i := 1 TO 6 DO
        nowpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
    END_FOR;

    FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
        Axis_now_pulse[i] := MC_RdAxisPar_FL(AxisNum := SINT_TO_USINT(AxesGroup_Axis_Num_In_Group[i - 1]),AddrOffset := 76);
        Axis_now_angel[i] := (Axis_now_pulse[i] - AxesGroup_JointOffsets[i]) * AxesGroup_JointDirections[i] / AxesGroup_JointGearRatios[i];
        Axis_last_angel[i] := Axis_now_angel[i];
    END_FOR;

    AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));
    BufCal_vs := AxesGroup_Now_TCP_Velocity * 1000;

    d := ESQR((nowpoint_tcp[1] - circle_path[1]) * (nowpoint_tcp[1] - circle_path[1]) + 
              (nowpoint_tcp[2] - circle_path[2]) * (nowpoint_tcp[2] - circle_path[2]) + 
              (nowpoint_tcp[3] - circle_path[3]) * (nowpoint_tcp[3] - circle_path[3]));
    now_s := 2.0 * radius * EASIN(d / radius / 2);

    AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
    IF AxesGroup_Run_Number = 0 OR AxesGroup_Interrupt = 1 THEN
        BufCal_dec := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 228(* AG_ID *));
        BufCal_jerk := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 232(* AG_IJ *));
    ELSE
        BufCal_dec := dec;
        BufCal_jerk := jerk;
    END_IF;
    BufCal_vel := BufCal_vs;
    BufCal_acc := acc;

    pointbuf_dtime[1] := 0;    // 将pointbuf清空

    IF BufCal_jerk > 0 THEN    // S型
        FB_S_Type_Motion_Par_Calc_3_Segments1(vs := BufCal_vs,vt := 0,acc := 0,dec := BufCal_dec,jerk := BufCal_jerk,t1 => ,t2 => ,t3 => ,s1 => ,s2 => ,s3 => ,t_3seg => stop_time,s_3seg => stop_s) ;
    ELSE                       // 梯形
        stop_time := BufCal_vs / BufCal_dec;
        stop_s := BufCal_vs * stop_time - BufCal_dec * stop_time * stop_time / 2.0;
    END_IF;

    FOR i := 99 TO 2 BY -1 DO
        IF path_s[i] >= now_s THEN
            path_point_start := i;
        END_IF;
    END_FOR;
    now_s := path_s[path_point_start];

    FOR i := 99 TO 2 BY -1 DO
        IF path_s[i] >= now_s + stop_s THEN
            path_point_end := i;
        END_IF;
    END_FOR;
    path_point_num := path_point_end - path_point_start + 1;

    IF debug_print_PATH THEN
        debug_printl_real1(0);
        debug_printt_real1(path_point_start); debug_printt_real1(path_point_end); debug_printt_real1(BufCal_acc); debug_printt_real1(stop_time); debug_printl_real1(stop_s);
        debug_printt_real1(BufCal_vs); debug_printt_real1(BufCal_vel); debug_printt_real1(BufCal_acc); debug_printt_real1(BufCal_dec); debug_printl_real1(BufCal_jerk);
    END_IF;

    FB_CircularPointBufCal1(circle_path := circle_path[6 * (path_point_start - 1) + 1], center_point := center_point[1],path_s := path_s[path_point_start],
                            path_point_num := path_point_num,pathchoice := pathchoice,
                            vs := BufCal_vs,vel := BufCal_vel,acc := BufCal_acc,dec := BufCal_dec,jerk := BufCal_jerk,
                            pointbuf_tcp => pointbuf_tcp[1],pointbuf_dtime => pointbuf_dtime[1],pointnumber => pointnumber);

    FOR i := 0 TO pointnumber - 1 DO
        FB_MCS_ACS_KinTransform1(axesgroup_kin := AxesGroup_Kin_Transform,TCP_TargetPos := ADR(pointbuf_tcp[6 * i + 1]),DenHartA := ADR(AxesGroup_DenHartA[1]),DenHartD := ADR(AxesGroup_DenHartD[1]),JointLimits := ADR(AxesGroup_JointLimits[1]),JointAngles_now := ADR(Axis_last_angel[1]),JointAngle1 => pointbuf_acp[8 * i + 1],JointAngle2 => pointbuf_acp[8 * i + 2],JointAngle3 => pointbuf_acp[8 * i + 3],JointAngle4 => pointbuf_acp[8 * i + 4],JointAngle5 => pointbuf_acp[8 * i + 5],JointAngle6 => pointbuf_acp[8 * i + 6],JointAngle7 => pointbuf_acp[8 * i + 7],JointAngle8 => pointbuf_acp[8 * i + 8],Error => Error,ErrorID => );

        FOR j := 1 TO AxesGroup_Axis_Sum_In_Group DO
            Axis_last_angel[j] := pointbuf_acp[8 * i + j];
        END_FOR;

        IF Error THEN
            ErrorID := 9000 + i;
            EXIT;
        END_IF;
    END_FOR;

    FOR i := 1 TO pointnumber DO
        pointbuf_pus1[i] := pointbuf_acp[8 * (i - 1) + 1] * AxesGroup_JointGearRatios[1] * AxesGroup_JointDirections[1] + AxesGroup_JointOffsets[1];
        pointbuf_pus2[i] := pointbuf_acp[8 * (i - 1) + 2] * AxesGroup_JointGearRatios[2] * AxesGroup_JointDirections[2] + AxesGroup_JointOffsets[2];
        pointbuf_pus3[i] := pointbuf_acp[8 * (i - 1) + 3] * AxesGroup_JointGearRatios[3] * AxesGroup_JointDirections[3] + AxesGroup_JointOffsets[3];
        pointbuf_pus4[i] := pointbuf_acp[8 * (i - 1) + 4] * AxesGroup_JointGearRatios[4] * AxesGroup_JointDirections[4] + AxesGroup_JointOffsets[4];
        pointbuf_pus5[i] := pointbuf_acp[8 * (i - 1) + 5] * AxesGroup_JointGearRatios[5] * AxesGroup_JointDirections[5] + AxesGroup_JointOffsets[5];
        pointbuf_pus6[i] := pointbuf_acp[8 * (i - 1) + 6] * AxesGroup_JointGearRatios[6] * AxesGroup_JointDirections[6] + AxesGroup_JointOffsets[6];
        pointbuf_pus7[i] := pointbuf_acp[8 * (i - 1) + 7] * AxesGroup_JointGearRatios[7] * AxesGroup_JointDirections[7] + AxesGroup_JointOffsets[7];
        pointbuf_pus8[i] := pointbuf_acp[8 * (i - 1) + 8] * AxesGroup_JointGearRatios[8] * AxesGroup_JointDirections[8] + AxesGroup_JointOffsets[8];
    END_FOR;

    IF pointnumber = 0 THEN
        Error := TRUE;
        ErrorID := 16#15(* ER_NoPathPoint *);

        debug_printl_real1(-UINT_TO_REAL(ErrorID));
        debug_printt_real1(nowpoint_tcp[1]); debug_printt_real1(nowpoint_tcp[2]); debug_printt_real1(nowpoint_tcp[3]); debug_printt_real1(nowpoint_tcp[4]); debug_printt_real1(nowpoint_tcp[5]); debug_printl_real1(nowpoint_tcp[6]);
        debug_printt_real1(auxpoint_tcp[1]); debug_printt_real1(auxpoint_tcp[2]); debug_printt_real1(auxpoint_tcp[3]); debug_printt_real1(auxpoint_tcp[4]); debug_printt_real1(auxpoint_tcp[5]); debug_printl_real1(auxpoint_tcp[6]);
        debug_printt_real1(endpoint_tcp[1]); debug_printt_real1(endpoint_tcp[2]); debug_printt_real1(endpoint_tcp[3]); debug_printt_real1(endpoint_tcp[4]); debug_printt_real1(endpoint_tcp[5]); debug_printl_real1(endpoint_tcp[6]);
    END_IF;

    FOR i := 1 TO pointnumber DO
        IF pointbuf_dtime[i] <= 0.00001 THEN
            Error := TRUE;
            ErrorID := 9898;
        END_IF;

        IF Error THEN
            debug_printl_real1(0);
            debug_printt_real1(path_point_start); debug_printt_real1(path_point_end); debug_printt_real1(BufCal_acc); debug_printt_real1(stop_time); debug_printl_real1(stop_s);
            debug_printt_real1(BufCal_vs); debug_printt_real1(BufCal_vel); debug_printt_real1(BufCal_acc); debug_printt_real1(BufCal_dec); debug_printl_real1(BufCal_jerk);
            debug_printl_real1(0);
            debug_printt_real1(nowpoint_tcp[1]); debug_printt_real1(nowpoint_tcp[2]); debug_printt_real1(nowpoint_tcp[3]); debug_printt_real1(nowpoint_tcp[4]); debug_printt_real1(nowpoint_tcp[5]); debug_printl_real1(nowpoint_tcp[6]);
            debug_printt_real1(auxpoint_tcp[1]); debug_printt_real1(auxpoint_tcp[2]); debug_printt_real1(auxpoint_tcp[3]); debug_printt_real1(auxpoint_tcp[4]); debug_printt_real1(auxpoint_tcp[5]); debug_printl_real1(auxpoint_tcp[6]);
            debug_printt_real1(endpoint_tcp[1]); debug_printt_real1(endpoint_tcp[2]); debug_printt_real1(endpoint_tcp[3]); debug_printt_real1(endpoint_tcp[4]); debug_printt_real1(endpoint_tcp[5]); debug_printl_real1(endpoint_tcp[6]);
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO 6 DO
                    debug_printt_real1(pointbuf_tcp[6 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_dtime[i]);
            END_FOR;
        END_IF;
    END_FOR;

    IF NOT Error THEN
        IF AxesGroup_Interrupt = 1 THEN
            stage := 8(* FBS_Inerrupt *);
        ELSE
            stage := 7(* FBS_RunBlend *);
        END_IF;

        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            pp_blend_exe[i] := TRUE;
        END_FOR;

        IF debug_print_TCP THEN
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO 6 DO
                    debug_printt_real1(pointbuf_tcp[6 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_dtime[i]);
            END_FOR;
        END_IF;

        IF debug_print_ACP THEN
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO AxesGroup_Axis_Sum_In_Group - 1 DO
                    debug_printt_real1(pointbuf_acp[8 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_acp[8 * (i - 1) + j]);
            END_FOR;
        END_IF;
    ELSE
        stage := 0(* FBS_WaitExec *);
    END_IF;
END_IF;

IF stage = 6(* FBS_CalcBlend *) THEN
    AxesGroup_Buffer_Busy := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *));
END_IF;

MC_PP_Blend1(Axis_num := axesgroup + 0,Execute := pp_blend_exe[1],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus1[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don1,Busy => pp_blend_bus1,Active => pp_blend_act1,CommandAborted => pp_blend_abr1,Error => pp_blend_err1,ErrorID => pp_blend_eid1) ;
MC_PP_Blend2(Axis_num := axesgroup + 1,Execute := pp_blend_exe[2],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus2[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don2,Busy => pp_blend_bus2,Active => pp_blend_act2,CommandAborted => pp_blend_abr2,Error => pp_blend_err2,ErrorID => pp_blend_eid2) ;
MC_PP_Blend3(Axis_num := axesgroup + 2,Execute := pp_blend_exe[3],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus3[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don3,Busy => pp_blend_bus3,Active => pp_blend_act3,CommandAborted => pp_blend_abr3,Error => pp_blend_err3,ErrorID => pp_blend_eid3) ;
MC_PP_Blend4(Axis_num := axesgroup + 3,Execute := pp_blend_exe[4],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus4[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don4,Busy => pp_blend_bus4,Active => pp_blend_act4,CommandAborted => pp_blend_abr4,Error => pp_blend_err4,ErrorID => pp_blend_eid4) ;
MC_PP_Blend5(Axis_num := axesgroup + 4,Execute := pp_blend_exe[5],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus5[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don5,Busy => pp_blend_bus5,Active => pp_blend_act5,CommandAborted => pp_blend_abr5,Error => pp_blend_err5,ErrorID => pp_blend_eid5) ;
MC_PP_Blend6(Axis_num := axesgroup + 5,Execute := pp_blend_exe[6],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus6[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don6,Busy => pp_blend_bus6,Active => pp_blend_act6,CommandAborted => pp_blend_abr6,Error => pp_blend_err6,ErrorID => pp_blend_eid6) ;
MC_PP_Blend7(Axis_num := axesgroup + 6,Execute := pp_blend_exe[7],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus7[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don7,Busy => pp_blend_bus7,Active => pp_blend_act7,CommandAborted => pp_blend_abr7,Error => pp_blend_err7,ErrorID => pp_blend_eid7) ;
MC_PP_Blend8(Axis_num := axesgroup + 7,Execute := pp_blend_exe[8],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus8[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don8,Busy => pp_blend_bus8,Active => pp_blend_act8,CommandAborted => pp_blend_abr8,Error => pp_blend_err8,ErrorID => pp_blend_eid8) ;

pp_blend_don[1] := pp_blend_don1; pp_blend_don[2] := pp_blend_don2; pp_blend_don[3] := pp_blend_don3; pp_blend_don[4] := pp_blend_don4; pp_blend_don[5] := pp_blend_don5; pp_blend_don[6] := pp_blend_don6; pp_blend_don[7] := pp_blend_don7; pp_blend_don[8] := pp_blend_don8;
pp_blend_bus[1] := pp_blend_bus1; pp_blend_bus[2] := pp_blend_bus2; pp_blend_bus[3] := pp_blend_bus3; pp_blend_bus[4] := pp_blend_bus4; pp_blend_bus[5] := pp_blend_bus5; pp_blend_bus[6] := pp_blend_bus6; pp_blend_bus[7] := pp_blend_bus7; pp_blend_bus[8] := pp_blend_bus8;
pp_blend_act[1] := pp_blend_act1; pp_blend_act[2] := pp_blend_act2; pp_blend_act[3] := pp_blend_act3; pp_blend_act[4] := pp_blend_act4; pp_blend_act[5] := pp_blend_act5; pp_blend_act[6] := pp_blend_act6; pp_blend_act[7] := pp_blend_act7; pp_blend_act[8] := pp_blend_act8;
pp_blend_abr[1] := pp_blend_abr1; pp_blend_abr[2] := pp_blend_abr2; pp_blend_abr[3] := pp_blend_abr3; pp_blend_abr[4] := pp_blend_abr4; pp_blend_abr[5] := pp_blend_abr5; pp_blend_abr[6] := pp_blend_abr6; pp_blend_abr[7] := pp_blend_abr7; pp_blend_abr[8] := pp_blend_abr8;

pp_blend_done := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_blend_done := pp_blend_done AND pp_blend_don[i];
END_FOR;
pp_blend_busy := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_blend_busy := pp_blend_busy AND pp_blend_bus[i];
END_FOR;
pp_blend_active_last := pp_blend_active;
pp_blend_active := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_blend_active := pp_blend_active AND pp_blend_act[i];
END_FOR;

IF stage = 7(* FBS_RunBlend *) THEN
    IF AxesGroup_Buffered_Transition_Mode = 0 THEN
        IF pp_blend_done THEN
            now_motion_done := TRUE;
        END_IF;
    ELSIF AxesGroup_Buffered_Transition_Mode = 1(* TM_StartV *) THEN
        AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));
        IF AxesGroup_Now_TCP_Velocity <> 0 AND AxesGroup_Now_TCP_Velocity * 1000 <= AxesGroup_Buffered_Transition_Parameter * blendVel THEN
            now_motion_done := TRUE;
        END_IF;
    ELSIF AxesGroup_Buffered_Transition_Mode = 2(* TM_ConstantV *) THEN
        AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));
        IF AxesGroup_Now_TCP_Velocity * 1000 <= AxesGroup_Buffered_Transition_Parameter * blendVel THEN
            now_motion_done := TRUE;
        END_IF;
    ELSIF AxesGroup_Buffered_Transition_Mode = 3(* TM_CornerDis *) THEN
        FOR i :=  1 TO 3 DO
            AxesGroup_Now_TCP_Position[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
        END_FOR;
        d := ESQR((AxesGroup_Now_TCP_Position[1] - endpoint_tcp[1]) * (AxesGroup_Now_TCP_Position[1] - endpoint_tcp[1]) +
                  (AxesGroup_Now_TCP_Position[2] - endpoint_tcp[2]) * (AxesGroup_Now_TCP_Position[2] - endpoint_tcp[2]) +
                  (AxesGroup_Now_TCP_Position[3] - endpoint_tcp[3]) * (AxesGroup_Now_TCP_Position[3] - endpoint_tcp[3]));
        s_to_end := 2.0 * radius * EASIN(d / radius / 2) + path_s[i - 1];
        IF s_to_end <= AxesGroup_Buffered_Transition_Parameter THEN
            now_motion_done := TRUE;
        END_IF;
    ELSIF AxesGroup_Buffered_Transition_Mode = 4(* TM_MaxCornerDeviation *) THEN
        FOR i :=  1 TO 3 DO
            AxesGroup_Now_TCP_Position[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
        END_FOR;
        d := ESQR((AxesGroup_Now_TCP_Position[1] - endpoint_tcp[1]) * (AxesGroup_Now_TCP_Position[1] - endpoint_tcp[1]) +
                  (AxesGroup_Now_TCP_Position[2] - endpoint_tcp[2]) * (AxesGroup_Now_TCP_Position[2] - endpoint_tcp[2]) +
                  (AxesGroup_Now_TCP_Position[3] - endpoint_tcp[3]) * (AxesGroup_Now_TCP_Position[3] - endpoint_tcp[3]));
        s_to_end := 2.0 * radius * EASIN(d / radius / 2) + path_s[i - 1];
        IF s_to_end <= AxesGroup_Buffered_Transition_Parameter THEN
            now_motion_done := TRUE;
        END_IF;
    END_IF;

    IF now_motion_done THEN
        now_motion_done := FALSE;
        Busy := FALSE;
        Active := FALSE;
        Done := NOT CommandAborted;

        stage := 0(* FBS_WaitExec *);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *), Data := 0);
    END_IF;
END_IF;

IF stage = 8(* FBS_Inerrupt *) THEN
    AxesGroup_Interrupt := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 422(* AG_TINR *));
    IF pp_blend_done THEN
        Active := FALSE;
        FOR i := 1 TO 6 DO
            pp_exe[i] := FALSE;
            pp_blend_exe[i] := FALSE;
        END_FOR;
    ELSIF AxesGroup_Interrupt = 0 AND NOT Active THEN
        BufCal_dec := dec;
        BufCal_jerk := jerk;
        stage := 3(* FBS_CalcRoute *);
    END_IF;
END_IF;

IF Error AND Active THEN
    IF pp_done OR pp_blend_done THEN
        Busy := FALSE;
        Active := FALSE;

        stage := 0(* FBS_WaitExec *);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *), Data := 0);
    END_IF;
END_IF;
lastExe := Execute;

    IF debug = 1 THEN
        AxesGroup := 7; CircMode := 1(* CM_Border *); PathChoice := 1(* PC_Clock *); CoordSystem := 2(* CS_MCS *);
        Velocity := 60; Acceleration := 20; Deceleration := 30; Jerk := 10;
        AuxPoint[1] := 200; AuxPoint[2] := 200; AuxPoint[3] := 0; AuxPoint[4] := 0; AuxPoint[5] := 0; AuxPoint[6] := 0;
        EndPoint[1] := 400; EndPoint[2] := 000; EndPoint[3] := 0; EndPoint[4] := 0; EndPoint[5] := 0; EndPoint[6] := 0;
        BufferMode := 0; TransitionMode := 0; TransitionParameter := 0;
        debug_print_PATH := TRUE; debug_print_TCP := TRUE;
    ELSIF debug = 2 THEN
        AxesGroup := 7; CircMode := 1(* CM_Border *); PathChoice := 1(* PC_Clock *); CoordSystem := 2(* CS_MCS *);
        Velocity := 60; Acceleration := 20; Deceleration := 30; Jerk := 0;
        AuxPoint[1] := 200; AuxPoint[2] := 200; AuxPoint[3] := 0; AuxPoint[4] := 0; AuxPoint[5] := 0; AuxPoint[6] := 0;
        EndPoint[1] := 400; EndPoint[2] := 000; EndPoint[3] := 0; EndPoint[4] := 0; EndPoint[5] := 0; EndPoint[6] := 0;
        BufferMode := 5; TransitionMode := 3; TransitionParameter := 100;
        debug_print_PATH := TRUE; debug_print_TCP := TRUE;
    END_IF;

END_PROGRAM

        debug : USINT;
        debug_print_TCP : BOOL;
        debug_print_ACP : BOOL;
        debug_print_PATH : BOOL;
        debug_printt_real1 : debug_printt_real;
        debug_printl_real1 : debug_printl_real;
END_VAR

VAR_INPUT
    AxesGroup : USINT;
    Execute : BOOL;
    CircMode : USINT;
    AuxPoint : ARRAY[1..8] OF REAL;
    EndPoint : ARRAY[1..8] OF REAL;
    PathChoice : USINT;
    Velocity : REAL;
    Acceleration : REAL;
    Deceleration : REAL;
    Jerk : REAL;
    CoordSystem : USINT;
    BufferMode : USINT;
    TransitionMode : USINT;
    TransitionParameter : REAL;
END_VAR

VAR_OUTPUT
    AxesGroupOut : USINT;
    Done : BOOL;
    Busy : BOOL;
    Active : BOOL;
    CommandAborted : BOOL;
    Error : BOOL;
    ErrorID : UINT;
END_VAR

IF stage = 0(* FBS_WaitExec *) AND NOT Execute AND NOT Active THEN
    Busy := FALSE;
    Active := FALSE;
    Done := FALSE;
    Error := FALSE;
    ErrorID := 0(* ER_NoError *);
    CommandAborted := FALSE;

    pp_done := FALSE;
    pp_blend_done := FALSE;
    FOR i := 1 TO 8 DO
        pp_exe[i] := FALSE;
        pp_blend_exe[i] := FALSE;
    END_FOR;
    group_RunNumberF := 0;
    pp_active_last := FALSE;
END_IF;

IF stage = 0(* FBS_WaitExec *) AND Execute AND NOT lastExe AND NOT Active THEN
    IF AxesGroup < 0(* AXESGROUP_MINNUM *) OR AxesGroup > 20(* AXESGROUP_MAXNUM *) THEN
        ErrorID := 16#05(* ER_AxesGroupNum_ParameterError *);
    END_IF;
    AxesGroup_State := MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 4(* AG_GS *));
    IF AxesGroup_State = 0(* GS_GroupDisabled *) OR AxesGroup_State = 1(* GS_GroupErrorstop *) THEN
        ErrorID := 16#01 (* ER_StateError *);
    ELSIF AxesGroup_State = 4(* GS_GroupStopping *) THEN
        ErrorID := 16#20(* ER_AxisState_Stopping_Error *);
    END_IF;
    IF Velocity < 0 OR Acceleration <= 0 OR Deceleration <= 0 OR Jerk < 0 THEN
        ErrorID := 16#02(* ER_ParameterError *);
    END_IF;
    IF CoordSystem <> 1(* CS_ACS *) AND CoordSystem <> 2(* CS_MCS *) AND CoordSystem <> 3(* CS_PCS *) THEN
        ErrorID := 16#06(* ER_CoordSystem_ParameterError *);
    END_IF;
    IF BufferMode >= 6 THEN
        ErrorID := 16#02(* ER_ParameterError *);
    END_IF;
    IF TransitionMode >= 5 THEN
        ErrorID := 16#02(* ER_ParameterError *);
    END_IF;

    IF ErrorID <> 0(* ER_NoError *) THEN
        Error := TRUE;
    ELSE
        AxesGroup_Axis_Sum_In_Group := MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 5(* AG_ASUM *));
        AxesGroup_Kin_Transform := MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 14(* AG_KT *));
        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            AxesGroup_JointLimits[i * 2 - 1] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 756(* AG_JL *) + (i - 1) * 2 * 4);
            AxesGroup_JointLimits[i * 2] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 756(* AG_JL *) + (i * 2 - 1) * 4);
            AxesGroup_DenHartA[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 16(* AG_DHA *) + (i - 1) * 4);
            AxesGroup_DenHartD[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 52(* AG_DHD *) + (i - 1) * 4);
            AxesGroup_JointDirections[i] := USINT_TO_SINT(MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 88(* AG_JD *) + (i - 1)));
            AxesGroup_JointOffsets[i] := MC_RdAxesPar_D(AxesNum := AxesGroup,AddrOffset := 96(* AG_JO *) + (i - 1) * 4);
            AxesGroup_JointGearRatios[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 192(* AG_JGR *) + (i - 1) * 4);

            AxesGroup_Axis_Num_In_Group[i - 1] := USINT_TO_SINT(MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 6(* AG_ANUM *) + i - 1));
        END_FOR;
            AxesGroup_DenHartA[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 16(* AG_DHA *) + (i - 1) * 4);
            AxesGroup_DenHartD[i] := MC_RdAxesPar_FL(AxesNum := AxesGroup,AddrOffset := 52(* AG_DHD *) + (i - 1) * 4);

        AxesGroupOut := AxesGroup;
        axesgroup := AxesGroup;
        vel := Velocity;
        acc := Acceleration;
        dec := Deceleration;
        jerk := Jerk;
        bufferMode := BufferMode;
        transitionMode := TransitionMode;
        transitionParameter := TransitionParameter;
        circMode := CircMode;
        pathchoice := PathChoice;
        coordSystem := CoordSystem;
        FOR i := 1 TO 6 DO
            endpoint[i] := EndPoint[i];
            auxpoint[i] := AuxPoint[i];
        END_FOR;

        AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
        group_RunNumberF := AxesGroup_Run_Number;

        Busy := TRUE;
        stage := 1(* FBS_WaitBuffer *);
    END_IF;
END_IF;

IF stage = 1(* FBS_WaitBuffer *) OR stage = 2(* FBS_InBuffer *) THEN
    AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
    IF AxesGroup_Run_Number <> group_RunNumberF THEN
        CommandAborted := TRUE;
        Busy := FALSE;
        stage := 0(* FBS_WaitExec *);
    END_IF;
END_IF;

IF stage = 1(* FBS_WaitBuffer *) THEN
    IF bufferMode = 0(* BM_Aborting *) THEN
        AxesGroup_Buffered_Transition_Mode := transitionMode;
        CASE AxesGroup_Buffered_Transition_Mode OF
            0(* TM_None *) :  ;
            1(* TM_StartV *) ,
            2(* TM_ConstantV *) : AxesGroup_Buffered_Transition_Parameter := transitionParameter;
            3(* TM_CornerDis *),
            4(* TM_MaxCornerDeviation *) : ErrorID := 16#27(* ER_Buf_Trans_Combination_NotSupported *);
        END_CASE;

        IF ErrorID = 0 THEN
            group_RunNumberF := AxesGroup_Run_Number MOD 255 + 1; // [1,255]
            WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *),Data := group_RunNumberF);
        END_IF;
    ELSE
        AxesGroup_Buffer_Busy := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *));
        IF AxesGroup_Buffer_Busy = 0 THEN
            AxesGroup_Buffered_Transition_Mode := transitionMode;
            CASE bufferMode OF
                1(* BM_Buffered *) :
                    CASE AxesGroup_Buffered_Transition_Mode OF
                        0(* TM_None *) :  ;
                        1(* TM_StartV *) ,
                        2(* TM_ConstantV *) ,
                        3(* TM_CornerDis *),
                        4(* TM_MaxCornerDeviation *) : ErrorID := 16#27(* ER_Buf_Trans_Combination_NotSupported *);
                    END_CASE;
                2(* BM_Blendinglow *) ,
                3(* BM_BlendingPrevious *) ,
                4(* BM_BlendingNext *) ,
                5(* BM_BlendingHigh *) :
                    CASE AxesGroup_Buffered_Transition_Mode OF
                        0(* TM_None *) : ErrorID := 16#27(* ER_Buf_Trans_Combination_NotSupported *);
                        1(* TM_StartV *) ,
                        2(* TM_ConstantV *) ,
                        3(* TM_CornerDis *) ,
                        4(* TM_MaxCornerDeviation *) : AxesGroup_Buffered_Transition_Parameter := transitionParameter;
                    END_CASE;
            END_CASE;
        END_IF;
    END_IF;

    IF ErrorID = 0(* ER_NoError *) THEN
        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *),Data := bufferMode);
        WRParRes := MC_WrAxesPar_FL(AxesNum := axesgroup,AddrOffset := 236(* AG_BTV *), Data := vel);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 421(* AG_BTM *),Data := AxesGroup_Buffered_Transition_Mode);
        WRParRes := MC_WrAxesPar_FL(AxesNum := axesgroup,AddrOffset := 224(* AG_BTP *),Data := AxesGroup_Buffered_Transition_Parameter);

        stage := 2(* FBS_InBuffer *);
    ELSE
        Error := TRUE;
        stage := 0(* FBS_WaitExec *);
    END_IF;
END_IF;

IF stage = 2(* FBS_InBuffer *) THEN
    AxesGroup_Busy := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *));
    IF AxesGroup_Busy = 0 THEN
        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *),Data := 0); 
        AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));

        IF transitionMode = 0 AND AxesGroup_Now_TCP_Velocity > 1 THEN
            Error := TRUE;
            ErrorID := 16#0B(* ER_Transition_Compute_Error *);
            stage := 0(* FBS_WaitExec *);
        ELSE
            stage := 3(* FBS_CalcRoute *);
        END_IF;
    END_IF;
END_IF;

IF stage = 3(* FBS_CalcRoute *) THEN
    IF bufferMode = 0 (* BM_Aborting *) THEN
        FOR i := 1 TO 6 DO
            nowpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
            strpoint_tcp[i] := nowpoint_tcp[i];
        END_FOR;

        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            Axis_now_pulse[i] := MC_RdAxisPar_FL(AxisNum := SINT_TO_USINT(AxesGroup_Axis_Num_In_Group[i - 1]),AddrOffset := 76);
            Axis_now_angel[i] := (Axis_now_pulse[i] - AxesGroup_JointOffsets[i]) * AxesGroup_JointDirections[i] / AxesGroup_JointGearRatios[i];
            Axis_last_angel[i] := Axis_now_angel[i];
        END_FOR;
    ELSE
        FOR i := 1 TO 6 DO
            nowpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
            strpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 668(* AG_MAP *) + 4 * (i - 1));
        END_FOR;

        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            AxesGroup_Now_Axis_Motion_AimPulse[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 284(* AG_NMA *) + (i - 1) * 4);
            Axis_last_angel[i] := (AxesGroup_Now_Axis_Motion_AimPulse[i] - AxesGroup_JointOffsets[i]) * AxesGroup_JointDirections[i] / AxesGroup_JointGearRatios[i];
        END_FOR;
    END_IF;

    IF coordSystem = 1(* CS_ACS *) THEN
        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            endpoint_tcp[i] := endpoint[i] + Axis_now_angel[i];
            auxpoint_tcp[i] := auxpoint[i] + Axis_now_angel[i];
        END_FOR;
        FB_ACS_MCS_KinTransform1(axesgroup_kin := AxesGroup_Kin_Transform,ACS_JointAngle := endpoint_tcp[1],DenHartA := AxesGroup_DenHartA[1],DenHartD := AxesGroup_DenHartD[1],MCS_X => endpoint_tcp[1],MCS_Y => endpoint_tcp[2],MCS_Z => endpoint_tcp[3],MCS_RX => endpoint_tcp[4],MCS_RY => endpoint_tcp[5],MCS_RZ => endpoint_tcp[6]); 
        FB_ACS_MCS_KinTransform1(axesgroup_kin := AxesGroup_Kin_Transform,ACS_JointAngle := auxpoint_tcp[1],DenHartA := AxesGroup_DenHartA[1],DenHartD := AxesGroup_DenHartD[1],MCS_X => auxpoint_tcp[1],MCS_Y => auxpoint_tcp[2],MCS_Z => auxpoint_tcp[3],MCS_RX => auxpoint_tcp[4],MCS_RY => auxpoint_tcp[5],MCS_RZ => auxpoint_tcp[6]); 
    ELSIF  coordSystem = 2(* CS_MCS *) THEN
        FOR i := 1 TO 6 DO
            endpoint_tcp[i] := endpoint[i] + strpoint_tcp[i];
            auxpoint_tcp[i] := auxpoint[i] + strpoint_tcp[i];
        END_FOR;
    END_IF;

    IF debug_print_PATH THEN
        debug_printl_real1(0);
        debug_printt_real1(nowpoint_tcp[1]); debug_printt_real1(nowpoint_tcp[2]); debug_printt_real1(nowpoint_tcp[3]); debug_printt_real1(nowpoint_tcp[4]); debug_printt_real1(nowpoint_tcp[5]); debug_printl_real1(nowpoint_tcp[6]);
        debug_printt_real1(auxpoint_tcp[1]); debug_printt_real1(auxpoint_tcp[2]); debug_printt_real1(auxpoint_tcp[3]); debug_printt_real1(auxpoint_tcp[4]); debug_printt_real1(auxpoint_tcp[5]); debug_printl_real1(auxpoint_tcp[6]);
        debug_printt_real1(endpoint_tcp[1]); debug_printt_real1(endpoint_tcp[2]); debug_printt_real1(endpoint_tcp[3]); debug_printt_real1(endpoint_tcp[4]); debug_printt_real1(endpoint_tcp[5]); debug_printl_real1(endpoint_tcp[6]);
    END_IF;

    FOR i := 1 TO 6 DO
        circle_path[i] := strpoint_tcp[i];
    END_FOR;

    FB_CircularPointsCal1(CircMode := circMode,PathChoice := pathchoice,
                          NowPosition := strpoint_tcp[1],AuxPosition := auxpoint_tcp[1],EndPosition := endpoint_tcp[1],PointNum := 99,
                          Error => cirerror,ErrorID => cirerrrorid,PointBuf => circle_path[7], CenterPoint => center_point[1]);

    IF circMode = 1(* CM_Border *) THEN
        circMode := 2(* CM_Center *);
        auxpoint_tcp[1] := center_point[1];
        auxpoint_tcp[2] := center_point[2];
        auxpoint_tcp[3] := center_point[3];
    END_IF;

    radius := ESQR((circle_path[1] - center_point[1]) * (circle_path[1] - center_point[1]) +
                   (circle_path[2] - center_point[2]) * (circle_path[2] - center_point[2]) +
                   (circle_path[3] - center_point[3]) * (circle_path[3] - center_point[3]));

    path_s[1] := 0;
    FOR i := 2 TO 100 DO
        d := ESQR((circle_path[6 * (i - 1) + 1] - circle_path[6 * (i - 2) + 1]) * (circle_path[6 * (i - 1) + 1] - circle_path[6 * (i - 2) + 1]) +
                  (circle_path[6 * (i - 1) + 2] - circle_path[6 * (i - 2) + 2]) * (circle_path[6 * (i - 1) + 2] - circle_path[6 * (i - 2) + 2]) +
                  (circle_path[6 * (i - 1) + 3] - circle_path[6 * (i - 2) + 3]) * (circle_path[6 * (i - 1) + 3] - circle_path[6 * (i - 2) + 3]));
        path_s[i] := 2.0 * radius * EASIN(d / radius / 2) + path_s[i - 1];
    END_FOR;

    IF debug_print_PATH THEN
        debug_printl_real1(0);
        FOR i := 1 TO 100 DO
            FOR j := 1 TO 6 DO
                debug_printt_real1(circle_path[6 * (i - 1) + j]);
            END_FOR;
                debug_printl_real1(path_s[i]);
        END_FOR;
    END_IF;

    pointbuf_dtime[1] := 0;    // 将pointbuf清空

    AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));

    IF transitionMode <> 0 AND AxesGroup_Now_TCP_Velocity <> 0 THEN
        FOR i := 1 TO 3 DO
            AxesGroup_Now_TCP_Direction[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 652(* AG_NTD *) + 4 * (i - 1));
        END_FOR;

        IF bufferMode > 1 THEN
            FOR i := 1 TO 6 DO
                strpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 668(* AG_MAP *) + 4 * (i - 1));
            END_FOR;
        END_IF;

        CASE transitionMode OF
            1(* TM_StartV *) :
                IF jerk = 0 THEN
                    curve_time := vel / acc;
                    trans_s := Motion_Delta_S(vs := 0,acc := acc,dec := 0,jerk := 0,d_t := curve_time);
                ELSE
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := 0,vt := vel,acc := acc,dec := 0,jerk := jerk,t1 => t1,t2 => t2,t3 => t3,s1 => s1,s2 => s2,s3 => s3,t_3seg => curve_time,s_3seg => trans_s);
                END_IF;

                FOR i := 1 TO 100 DO
                    IF path_s[i] >= trans_s THEN
                        tendpoint_index := i;
                        EXIT;
                    END_IF;
                END_FOR;
                curve_time := curve_time + (path_s[tendpoint_index] - trans_s) / vel;

                svel := AxesGroup_Now_TCP_Velocity * 1000; evel := vel;
            2(* TM_ConstantV *) :
                IF jerk = 0 THEN
                    curve_time := (vel * transitionParameter) / acc;
                    trans_s := Motion_Delta_S(vs := 0,acc := acc,dec := 0,jerk := 0,d_t := curve_time);
                ELSE
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := 0,vt := vel,acc := acc,dec := 0,jerk := jerk,t1 => t1,t2 => t2,t3 => t3,s1 => s1,s2 => s2,s3 => s3,t_3seg => ,s_3seg => );
                    stage_v1 := 0;
                    stage_v2 := 0.5 * jerk * t1 * t1;
                    stage_v3 := stage_v2 + jerk * t1 * t2;
                    IF 0.5 * jerk * t1 * t1 < vel * transitionParameter THEN
                        curve_time := ESQR(vel * transitionParameter * 2 / jerk);
                        trans_s := Motion_Delta_S(vs := 0,acc := 0,dec := 0,jerk := jerk,d_t := ESQR(vel * transitionParameter * 2 / jerk));
                    ELSIF 0.5 * jerk * t1 * t1 + acc * t2 < vel * transitionParameter THEN
                        trans_s := s1 + Motion_Delta_S(vs := stage_v2,acc := acc,dec := 0,jerk := 0,d_t := (vel * transitionParameter - 0.5 * jerk * t1 * t1) / acc);
                        curve_time := (vel * transitionParameter - 0.5 * jerk * t1 * t1) / acc + t1;
                    ELSE
                        trans_s := s1 + s2 + Motion_Delta_S(vs := stage_v3,acc := acc,dec := 0,jerk := -jerk,d_t := (vel * transitionParameter - 0.5 * jerk * t1 * t1) / acc);
                        curve_time := (vel * transitionParameter - 0.5 * jerk * t1 * t1) / acc + t1 + t2;
                    END_IF;
                END_IF;

                FOR i := 1 TO 100 DO
                    IF path_s[i] >= trans_s THEN
                        tendpoint_index := i;
                        EXIT;
                    END_IF;
                END_FOR;
                curve_time := curve_time + (path_s[tendpoint_index] - trans_s) / (vel * transitionParameter);

                svel := AxesGroup_Now_TCP_Velocity * 1000; evel := svel;
            3(* TM_CornerDis *),
            4(* TM_MaxCornerDeviation *) :
                FOR i := 1 TO 100 DO
                    IF path_s[i] >= transitionParameter THEN
                        tendpoint_index := i;
                        EXIT;
                    END_IF;
                END_FOR;

                trans_s := path_s[i];
                IF jerk = 0 THEN
                    t1 := vel / acc;    // 开始速度为0
                    t3 := vel / dec;    // 结束速度为0
                    s1 := Motion_Delta_S(vs := 0,acc := acc,dec := 0,jerk := 0,d_t := t1);
                    s3 := Motion_Delta_S(vs := vel,acc := 0,dec := dec,jerk := 0,d_t := t3);
                    s2 := path_s[100] - s1 - s3;
                    t2 := s2 / vel;
                    vmax := vel;
                    IF t2 < 0 THEN
                        vmax := ESQR((2 * acc * dec * path_s[100]) / (acc + dec));
                        t1 := vmax / acc;    // 开始速度为0
                        t2 := 0;
                        t3 := vmax / dec;    // 结束速度为0
                        s1 := acc * t1 * t1 / 2;
                        s2 := 0;
                        s3 := path_s[100] - s1;
                    END_IF;

                    IF trans_s <= s1 THEN
                        curve_time := ESQR(2 * trans_s / acc);
                        evel := acc * curve_time;
                    ELSIF trans_s <= s1 + s2 THEN
                        curve_time := t1 + (trans_s - s1) / vmax;
                        evel := vmax;
                    ELSE
                        a := dec / 2.0;
                        b := -vmax;
                        c := trans_s - s1 - s2;
                        curve_time := (ESQR(b * b - 4 * a * c) - b) / (2 * a);
                        IF curve_time < 0 OR curve_time > t3 THEN
                            curve_time := ( -b - ESQR(b * b - 4 * a * c)) / (2 * a);
                        END_IF;
                        curve_time := t1 + t2 + curve_time;
                        evel := vmax - dec * curve_time;
                    END_IF;
                ELSE
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := 0,vt := vel,acc := acc,dec := 0,jerk := jerk,t1 => t1,t2 => t2,t3 => t3,s1 => s1,s2 => s2,s3 => s3,t_3seg => ,s_3seg => );
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := vel,vt := 0,acc := 0,dec := dec,jerk := jerk,t1 => t5,t2 => t6,t3 => t7,s1 => s5,s2 => s6,s3 => s7,t_3seg => ,s_3seg => );
                    s4 := path_s[100] - s1 - s2 - s3 - s5 - s6 - s7;
                    t4 := s4 / vel;
                    IF t4 > 0 THEN
                        stage_v1 := 0;
                        stage_v2 := 0.5 * jerk * t1 * t1;
                        stage_v3 := stage_v2 + jerk * t1 * t2;
                        IF trans_s < s1 THEN
                            curve_time := Solve_Cubic_Equation(a := jerk / 6.0,b := 0,c := 0,d := -trans_s, LimitL := 0, LimitR := t1);
                            evel := stage_v1 + 0.5 * jerk * curve_time * curve_time;
                        ELSIF trans_s < s1 + s2 THEN
                            trans_s := trans_s - s1;
                            IF - 0.001 < trans_s AND trans_s < 0.001 THEN trans_s := 0; END_IF;
                            a := acc / 2.0;
                            b := stage_v2;
                            c := -trans_s;
                            curve_time := (ESQR(b * b - 4 * a * c) - b) / (2 * a);
                            IF curve_time < 0 OR curve_time > t2 THEN
                                curve_time := ( -b - ESQR(b * b - 4 * a * c)) / (2 * a);
                            END_IF;
                            evel := stage_v2 + acc * curve_time;
                            curve_time := curve_time + t1;
                        ELSIF trans_s < s1 + s2 + s3 THEN
                            trans_s := trans_s - s1 - s2;
                            curve_time := Solve_Cubic_Equation(a := jerk / - 6.0,b := jerk * t1 / 2,c := stage_v3,d := -trans_s, LimitL := 0, LimitR := t3);
                            evel := stage_v3 + jerk * t1 * curve_time - 0.5 * jerk * curve_time * curve_time;
                            curve_time := curve_time + t1 + t2;
                        ELSIF trans_s < s1 + s2 + s3 + s4 THEN
                            trans_s := trans_s - s1 - s2 - s3;
                            curve_time := trans_s / vel;
                            evel := vel;
                            curve_time := curve_time + t1 + t2 + t3;
                        END_IF;
                        trans_s := path_s[i];
                    END_IF;
                END_IF;
                svel := AxesGroup_Now_TCP_Velocity * 1000;
        END_CASE;

        FOR i := 1 TO 100 DO
            IF path_s[i] >= trans_s THEN
                tendpoint_index := i;
                EXIT;
            END_IF;
        END_FOR;

        FOR i := 1 TO 6 DO
            tendpoint_tcp[i] := circle_path[6 * (tendpoint_index - 1) + i];
            tendpoint_dir[i] := (circle_path[6 * (tendpoint_index - 1) + i] - circle_path[6 * (tendpoint_index - 2) + i]) / (path_s[tendpoint_index] - path_s[tendpoint_index - 1]);
        END_FOR;

        FB_TransitionCurveCalc1(spoint_TCP := nowpoint_tcp[1], svel := svel,svel_dir := AxesGroup_Now_TCP_Direction[1],
                                epoint_TCP := tendpoint_tcp[1],evel := evel,evel_dir := tendpoint_dir[1],pointnumber := 20,
                                PointBuf_TCP => pointbuf_tcp[1],PointBuf_dtime => pointbuf_dtime[1]);
        pointbuf_dtime[20 + 1] := curve_time; pointbuf_dtime[20 + 2] := 0;

        IF debug_print_PATH THEN
            debug_printl_real1(0);
            FOR i := 1 TO 20 DO
                FOR j := 1 TO 6 DO
                    debug_printt_real1(pointbuf_tcp[6 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_dtime[i]);
            END_FOR;
        END_IF;
    END_IF;

    BufCal_vel := vel;
    BufCal_acc := acc;
    BufCal_dec := dec;
    BufCal_jerk := jerk;
    path_point_num := 100;

    FB_CircularPointBufCal1(circle_path := circle_path[1], center_point := center_point[1],path_s := path_s[1],path_point_num := path_point_num,pathchoice := pathchoice,
                            vs := 0,vel := BufCal_vel,acc := BufCal_acc,dec := BufCal_dec,jerk := BufCal_jerk,
                            pointbuf_tcp => pointbuf_tcp[1],pointbuf_dtime => pointbuf_dtime[1],pointnumber => pointnumber);

    FOR i := 0 TO pointnumber - 1 DO
        FB_MCS_ACS_KinTransform1(axesgroup_kin := AxesGroup_Kin_Transform,TCP_TargetPos := ADR(pointbuf_tcp[6 * i + 1]),DenHartA := ADR(AxesGroup_DenHartA[1]),DenHartD := ADR(AxesGroup_DenHartD[1]),JointLimits := ADR(AxesGroup_JointLimits[1]),JointAngles_now := ADR(Axis_last_angel[1]),JointAngle1 => pointbuf_acp[8 * i + 1],JointAngle2 => pointbuf_acp[8 * i + 2],JointAngle3 => pointbuf_acp[8 * i + 3],JointAngle4 => pointbuf_acp[8 * i + 4],JointAngle5 => pointbuf_acp[8 * i + 5],JointAngle6 => pointbuf_acp[8 * i + 6],JointAngle7 => pointbuf_acp[8 * i + 7],JointAngle8 => pointbuf_acp[8 * i + 8],Error => Error,ErrorID => );

        FOR j := 1 TO AxesGroup_Axis_Sum_In_Group DO
            Axis_last_angel[j] := pointbuf_acp[8 * i + j];
        END_FOR;

        IF Error THEN
            ErrorID := 9000 + i;
            EXIT;
        END_IF;
    END_FOR;

    FOR i := 1 TO pointnumber DO
        pointbuf_pus1[i] := pointbuf_acp[8 * (i - 1) + 1] * AxesGroup_JointGearRatios[1] * AxesGroup_JointDirections[1] + AxesGroup_JointOffsets[1];
        pointbuf_pus2[i] := pointbuf_acp[8 * (i - 1) + 2] * AxesGroup_JointGearRatios[2] * AxesGroup_JointDirections[2] + AxesGroup_JointOffsets[2];
        pointbuf_pus3[i] := pointbuf_acp[8 * (i - 1) + 3] * AxesGroup_JointGearRatios[3] * AxesGroup_JointDirections[3] + AxesGroup_JointOffsets[3];
        pointbuf_pus4[i] := pointbuf_acp[8 * (i - 1) + 4] * AxesGroup_JointGearRatios[4] * AxesGroup_JointDirections[4] + AxesGroup_JointOffsets[4];
        pointbuf_pus5[i] := pointbuf_acp[8 * (i - 1) + 5] * AxesGroup_JointGearRatios[5] * AxesGroup_JointDirections[5] + AxesGroup_JointOffsets[5];
        pointbuf_pus6[i] := pointbuf_acp[8 * (i - 1) + 6] * AxesGroup_JointGearRatios[6] * AxesGroup_JointDirections[6] + AxesGroup_JointOffsets[6];
        pointbuf_pus7[i] := pointbuf_acp[8 * (i - 1) + 7] * AxesGroup_JointGearRatios[7] * AxesGroup_JointDirections[7] + AxesGroup_JointOffsets[7];
        pointbuf_pus8[i] := pointbuf_acp[8 * (i - 1) + 8] * AxesGroup_JointGearRatios[8] * AxesGroup_JointDirections[8] + AxesGroup_JointOffsets[8];
    END_FOR;

    IF pointnumber = 0 THEN
        Error := TRUE;
        ErrorID := 16#15(* ER_NoPathPoint *);

        debug_printl_real1(-UINT_TO_REAL(ErrorID));
        debug_printt_real1(nowpoint_tcp[1]); debug_printt_real1(nowpoint_tcp[2]); debug_printt_real1(nowpoint_tcp[3]); debug_printt_real1(nowpoint_tcp[4]); debug_printt_real1(nowpoint_tcp[5]); debug_printl_real1(nowpoint_tcp[6]);
        debug_printt_real1(auxpoint_tcp[1]); debug_printt_real1(auxpoint_tcp[2]); debug_printt_real1(auxpoint_tcp[3]); debug_printt_real1(auxpoint_tcp[4]); debug_printt_real1(auxpoint_tcp[5]); debug_printl_real1(auxpoint_tcp[6]);
        debug_printt_real1(endpoint_tcp[1]); debug_printt_real1(endpoint_tcp[2]); debug_printt_real1(endpoint_tcp[3]); debug_printt_real1(endpoint_tcp[4]); debug_printt_real1(endpoint_tcp[5]); debug_printl_real1(endpoint_tcp[6]);
    ELSE
        FOR i := 1 TO pointnumber DO
            IF pointbuf_dtime[i] <= 0.0001 THEN
                Error := TRUE;
                ErrorID := 16#28(* ER_Path_Dtime_Zero *);
                EXIT;
            END_IF;
        END_FOR;
    END_IF;

    IF NOT Error THEN
        stage := 4(* FBS_Moving *);

        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            pp_exe[i] := TRUE;
        END_FOR;

        IF debug_print_TCP THEN
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO 6 DO
                    debug_printt_real1(pointbuf_tcp[6 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_dtime[i]);
            END_FOR;
        END_IF;

        IF debug_print_ACP THEN
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO AxesGroup_Axis_Sum_In_Group - 1 DO
                    debug_printt_real1(pointbuf_acp[8 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_acp[8 * (i - 1) + j]);
            END_FOR;
        END_IF;
    ELSE
        stage := 0(* FBS_WaitExec *);
    END_IF;
END_IF;

MC_PositionProfile1(Axis_num := axesgroup + 0,Execute := pp_exe[1],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus1[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don1,Busy => pp_bus1,Active => pp_act1,CommandAborted => pp_abr1,Error => pp_err1,ErrorID => pp_eid1) ;
MC_PositionProfile2(Axis_num := axesgroup + 1,Execute := pp_exe[2],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus2[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don2,Busy => pp_bus2,Active => pp_act2,CommandAborted => pp_abr2,Error => pp_err2,ErrorID => pp_eid2) ;
MC_PositionProfile3(Axis_num := axesgroup + 2,Execute := pp_exe[3],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus3[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don3,Busy => pp_bus3,Active => pp_act3,CommandAborted => pp_abr3,Error => pp_err3,ErrorID => pp_eid3) ;
MC_PositionProfile4(Axis_num := axesgroup + 3,Execute := pp_exe[4],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus4[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don4,Busy => pp_bus4,Active => pp_act4,CommandAborted => pp_abr4,Error => pp_err4,ErrorID => pp_eid4) ;
MC_PositionProfile5(Axis_num := axesgroup + 4,Execute := pp_exe[5],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus5[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don5,Busy => pp_bus5,Active => pp_act5,CommandAborted => pp_abr5,Error => pp_err5,ErrorID => pp_eid5) ;
MC_PositionProfile6(Axis_num := axesgroup + 5,Execute := pp_exe[6],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus6[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don6,Busy => pp_bus6,Active => pp_act6,CommandAborted => pp_abr6,Error => pp_err6,ErrorID => pp_eid6) ;
MC_PositionProfile7(Axis_num := axesgroup + 6,Execute := pp_exe[7],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus7[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don7,Busy => pp_bus7,Active => pp_act7,CommandAborted => pp_abr7,Error => pp_err7,ErrorID => pp_eid7) ;
MC_PositionProfile8(Axis_num := axesgroup + 7,Execute := pp_exe[8],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus8[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_don8,Busy => pp_bus8,Active => pp_act8,CommandAborted => pp_abr8,Error => pp_err8,ErrorID => pp_eid8) ;

pp_don[1] := pp_don1; pp_don[2] := pp_don2; pp_don[3] := pp_don3; pp_don[4] := pp_don4; pp_don[5] := pp_don5; pp_don[6] := pp_don6; pp_don[7] := pp_don7; pp_don[8] := pp_don8;
pp_bus[1] := pp_bus1; pp_bus[2] := pp_bus2; pp_bus[3] := pp_bus3; pp_bus[4] := pp_bus4; pp_bus[5] := pp_bus5; pp_bus[6] := pp_bus6; pp_bus[7] := pp_bus7; pp_bus[8] := pp_bus8;
pp_act[1] := pp_act1; pp_act[2] := pp_act2; pp_act[3] := pp_act3; pp_act[4] := pp_act4; pp_act[5] := pp_act5; pp_act[6] := pp_act6; pp_act[7] := pp_act7; pp_act[8] := pp_act8;
pp_abr[1] := pp_abr1; pp_abr[2] := pp_abr2; pp_abr[3] := pp_abr3; pp_abr[4] := pp_abr4; pp_abr[5] := pp_abr5; pp_abr[6] := pp_abr6; pp_abr[7] := pp_abr7; pp_abr[8] := pp_abr8;

pp_done := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_done := pp_done AND pp_don[i];
END_FOR;
pp_busy := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_busy := pp_busy AND pp_bus[i];
END_FOR;
pp_active := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_active := pp_active AND pp_act[i];
END_FOR;

IF stage = 4(* FBS_Moving *) THEN
    ;
END_IF;

IF stage = 4(* FBS_Moving *) THEN
    Active := TRUE;

    AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
    IF AxesGroup_Run_Number <> group_RunNumberF AND group_RunNumberF <> 0 THEN
        CommandAborted := TRUE;

        AxesGroup_Buffered_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 236(* AG_BTV *));
        AxesGroup_Buffered_Transition_Mode := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 421(* AG_BTM *));
        AxesGroup_Buffered_Transition_Parameter := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 224(* AG_BTP *));

        CASE AxesGroup_Buffered_Transition_Mode OF
            0(* TM_None *) : ;
            1(* TM_StartV *) : blendVel  := vel;
            2(* TM_ConstantV *) : blendVel := AxesGroup_Buffered_TCP_Velocity;
        END_CASE;

        stage := 5(* FBS_CalcAbort *);
    END_IF;

    AxesGroup_Interrupt := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 422(* AG_TINR *));
    IF AxesGroup_Interrupt = 1 THEN
        AxesGroup_Interrupt_Deceleration := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 228(* AG_ID *));
        AxesGroup_Interrupt_Jerk := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 232(* AG_IJ *));

        stage := 5(* FBS_CalcAbort *);
    END_IF;

    AxesGroup_Buffer_Busy := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *));
    IF AxesGroup_Buffer_Busy > 1 THEN
        AxesGroup_Buffered_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 236(* AG_BTV *));

        CASE AxesGroup_Buffer_Busy OF
            2(* BM_Blendinglow *) : IF vel < AxesGroup_Buffered_TCP_Velocity THEN blendVel := vel; ELSE blendVel := AxesGroup_Buffered_TCP_Velocity; END_IF;
            3(* BM_BlendingPrevious *) : blendVel := vel;
            4(* BM_BlendingNext *) : blendVel := AxesGroup_Buffered_TCP_Velocity;
            5(* BM_BlendingHigh *) : IF vel < AxesGroup_Buffered_TCP_Velocity THEN blendVel := AxesGroup_Buffered_TCP_Velocity; ELSE blendVel := vel; END_IF;
        END_CASE;

        AxesGroup_Buffered_Transition_Mode := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 421(* AG_BTM *));
        AxesGroup_Buffered_Transition_Parameter := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 224(* AG_BTP *));

        CASE AxesGroup_Buffered_Transition_Mode OF
            1(* TM_StartV *) ,  // 不重新规划轨迹，正常减速到过渡速度
            2(* TM_ConstantV *) : stage := 7(* FBS_RunBlend *); // 过渡速度由BufferMode和TransitionParameter共同决定，是否需要重新规划轨迹速度待考虑
            3(* TM_CornerDis *),
            4(* TM_MaxCornerDeviation *) : stage := 7(* FBS_CalcBlend *); // 先不重新规划速度曲线
        END_CASE;
    END_IF;

    IF pp_busy AND pp_active AND NOT pp_active_last THEN
        AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
        group_RunNumberF := AxesGroup_Run_Number MOD 255 + 1; // [1,255]
        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *),Data := group_RunNumberF);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *),Data := 0); 
        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *), Data := 1);
        AxesGroup_Now_Axis_Motion_AimPulse[1] := pointbuf_pus1[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[2] := pointbuf_pus2[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[3] := pointbuf_pus3[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[4] := pointbuf_pus4[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[5] := pointbuf_pus5[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[6] := pointbuf_pus6[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[7] := pointbuf_pus7[pointnumber];
        AxesGroup_Now_Axis_Motion_AimPulse[8] := pointbuf_pus8[pointnumber];
        FOR i := 1 TO 6 DO
            AxesGroup_TCP_Motion_Aimpos[i] := pointbuf_tcp[REAL_TO_UDINT(6 * (pointnumber - 1) + i)];
            WRParRes := MC_WrAxesPar_FL(AxesNum := axesgroup,AddrOffset := 668(* AG_MAP *) + (i - 1) * 4, Data := AxesGroup_TCP_Motion_Aimpos[i]);
            WRParRes := MC_WrAxesPar_FL(AxesNum := axesgroup,AddrOffset := 284(* AG_NMA *) + (i - 1) * 4, Data := AxesGroup_Now_Axis_Motion_AimPulse[i]);
        END_FOR;
    END_IF;
    pp_active_last := pp_active;

    IF pp_done THEN
        Done := TRUE;
        Error := FALSE;
        Busy := FALSE;
        Active := FALSE;
        stage := 0(* FBS_WaitExec *);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *), Data := 0);
    END_IF;
END_IF;

IF stage = 5(* FBS_CalcAbort *) THEN
    FOR i := 1 TO 6 DO
        nowpoint_tcp[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
    END_FOR;

    FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
        Axis_now_pulse[i] := MC_RdAxisPar_FL(AxisNum := SINT_TO_USINT(AxesGroup_Axis_Num_In_Group[i - 1]),AddrOffset := 76);
        Axis_now_angel[i] := (Axis_now_pulse[i] - AxesGroup_JointOffsets[i]) * AxesGroup_JointDirections[i] / AxesGroup_JointGearRatios[i];
        Axis_last_angel[i] := Axis_now_angel[i];
    END_FOR;

    AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));
    BufCal_vs := AxesGroup_Now_TCP_Velocity * 1000;

    d := ESQR((nowpoint_tcp[1] - circle_path[1]) * (nowpoint_tcp[1] - circle_path[1]) + 
              (nowpoint_tcp[2] - circle_path[2]) * (nowpoint_tcp[2] - circle_path[2]) + 
              (nowpoint_tcp[3] - circle_path[3]) * (nowpoint_tcp[3] - circle_path[3]));
    now_s := 2.0 * radius * EASIN(d / radius / 2);

    AxesGroup_Run_Number := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 415(* AG_RN *));
    IF AxesGroup_Run_Number = 0 OR AxesGroup_Interrupt = 1 THEN
        BufCal_dec := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 228(* AG_ID *));
        BufCal_jerk := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 232(* AG_IJ *));
    ELSE
        BufCal_dec := dec;
        BufCal_jerk := jerk;
    END_IF;
    BufCal_vel := BufCal_vs;
    BufCal_acc := acc;

    pointbuf_dtime[1] := 0;    // 将pointbuf清空

    IF BufCal_jerk > 0 THEN    // S型
        FB_S_Type_Motion_Par_Calc_3_Segments1(vs := BufCal_vs,vt := 0,acc := 0,dec := BufCal_dec,jerk := BufCal_jerk,t1 => ,t2 => ,t3 => ,s1 => ,s2 => ,s3 => ,t_3seg => stop_time,s_3seg => stop_s) ;
    ELSE                       // 梯形
        stop_time := BufCal_vs / BufCal_dec;
        stop_s := BufCal_vs * stop_time - BufCal_dec * stop_time * stop_time / 2.0;
    END_IF;

    FOR i := 99 TO 2 BY -1 DO
        IF path_s[i] >= now_s THEN
            path_point_start := i;
        END_IF;
    END_FOR;
    now_s := path_s[path_point_start];

    FOR i := 99 TO 2 BY -1 DO
        IF path_s[i] >= now_s + stop_s THEN
            path_point_end := i;
        END_IF;
    END_FOR;
    path_point_num := path_point_end - path_point_start + 1;

    IF debug_print_PATH THEN
        debug_printl_real1(0);
        debug_printt_real1(path_point_start); debug_printt_real1(path_point_end); debug_printt_real1(BufCal_acc); debug_printt_real1(stop_time); debug_printl_real1(stop_s);
        debug_printt_real1(BufCal_vs); debug_printt_real1(BufCal_vel); debug_printt_real1(BufCal_acc); debug_printt_real1(BufCal_dec); debug_printl_real1(BufCal_jerk);
    END_IF;

    FB_CircularPointBufCal1(circle_path := circle_path[6 * (path_point_start - 1) + 1], center_point := center_point[1],path_s := path_s[path_point_start],
                            path_point_num := path_point_num,pathchoice := pathchoice,
                            vs := BufCal_vs,vel := BufCal_vel,acc := BufCal_acc,dec := BufCal_dec,jerk := BufCal_jerk,
                            pointbuf_tcp => pointbuf_tcp[1],pointbuf_dtime => pointbuf_dtime[1],pointnumber => pointnumber);

    FOR i := 0 TO pointnumber - 1 DO
        FB_MCS_ACS_KinTransform1(axesgroup_kin := AxesGroup_Kin_Transform,TCP_TargetPos := ADR(pointbuf_tcp[6 * i + 1]),DenHartA := ADR(AxesGroup_DenHartA[1]),DenHartD := ADR(AxesGroup_DenHartD[1]),JointLimits := ADR(AxesGroup_JointLimits[1]),JointAngles_now := ADR(Axis_last_angel[1]),JointAngle1 => pointbuf_acp[8 * i + 1],JointAngle2 => pointbuf_acp[8 * i + 2],JointAngle3 => pointbuf_acp[8 * i + 3],JointAngle4 => pointbuf_acp[8 * i + 4],JointAngle5 => pointbuf_acp[8 * i + 5],JointAngle6 => pointbuf_acp[8 * i + 6],JointAngle7 => pointbuf_acp[8 * i + 7],JointAngle8 => pointbuf_acp[8 * i + 8],Error => Error,ErrorID => );

        FOR j := 1 TO AxesGroup_Axis_Sum_In_Group DO
            Axis_last_angel[j] := pointbuf_acp[8 * i + j];
        END_FOR;

        IF Error THEN
            ErrorID := 9000 + i;
            EXIT;
        END_IF;
    END_FOR;

    FOR i := 1 TO pointnumber DO
        pointbuf_pus1[i] := pointbuf_acp[8 * (i - 1) + 1] * AxesGroup_JointGearRatios[1] * AxesGroup_JointDirections[1] + AxesGroup_JointOffsets[1];
        pointbuf_pus2[i] := pointbuf_acp[8 * (i - 1) + 2] * AxesGroup_JointGearRatios[2] * AxesGroup_JointDirections[2] + AxesGroup_JointOffsets[2];
        pointbuf_pus3[i] := pointbuf_acp[8 * (i - 1) + 3] * AxesGroup_JointGearRatios[3] * AxesGroup_JointDirections[3] + AxesGroup_JointOffsets[3];
        pointbuf_pus4[i] := pointbuf_acp[8 * (i - 1) + 4] * AxesGroup_JointGearRatios[4] * AxesGroup_JointDirections[4] + AxesGroup_JointOffsets[4];
        pointbuf_pus5[i] := pointbuf_acp[8 * (i - 1) + 5] * AxesGroup_JointGearRatios[5] * AxesGroup_JointDirections[5] + AxesGroup_JointOffsets[5];
        pointbuf_pus6[i] := pointbuf_acp[8 * (i - 1) + 6] * AxesGroup_JointGearRatios[6] * AxesGroup_JointDirections[6] + AxesGroup_JointOffsets[6];
        pointbuf_pus7[i] := pointbuf_acp[8 * (i - 1) + 7] * AxesGroup_JointGearRatios[7] * AxesGroup_JointDirections[7] + AxesGroup_JointOffsets[7];
        pointbuf_pus8[i] := pointbuf_acp[8 * (i - 1) + 8] * AxesGroup_JointGearRatios[8] * AxesGroup_JointDirections[8] + AxesGroup_JointOffsets[8];
    END_FOR;

    IF pointnumber = 0 THEN
        Error := TRUE;
        ErrorID := 16#15(* ER_NoPathPoint *);

        debug_printl_real1(-UINT_TO_REAL(ErrorID));
        debug_printt_real1(nowpoint_tcp[1]); debug_printt_real1(nowpoint_tcp[2]); debug_printt_real1(nowpoint_tcp[3]); debug_printt_real1(nowpoint_tcp[4]); debug_printt_real1(nowpoint_tcp[5]); debug_printl_real1(nowpoint_tcp[6]);
        debug_printt_real1(auxpoint_tcp[1]); debug_printt_real1(auxpoint_tcp[2]); debug_printt_real1(auxpoint_tcp[3]); debug_printt_real1(auxpoint_tcp[4]); debug_printt_real1(auxpoint_tcp[5]); debug_printl_real1(auxpoint_tcp[6]);
        debug_printt_real1(endpoint_tcp[1]); debug_printt_real1(endpoint_tcp[2]); debug_printt_real1(endpoint_tcp[3]); debug_printt_real1(endpoint_tcp[4]); debug_printt_real1(endpoint_tcp[5]); debug_printl_real1(endpoint_tcp[6]);
    END_IF;

    FOR i := 1 TO pointnumber DO
        IF pointbuf_dtime[i] <= 0.00001 THEN
            Error := TRUE;
            ErrorID := 9898;
        END_IF;

        IF Error THEN
            debug_printl_real1(0);
            debug_printt_real1(path_point_start); debug_printt_real1(path_point_end); debug_printt_real1(BufCal_acc); debug_printt_real1(stop_time); debug_printl_real1(stop_s);
            debug_printt_real1(BufCal_vs); debug_printt_real1(BufCal_vel); debug_printt_real1(BufCal_acc); debug_printt_real1(BufCal_dec); debug_printl_real1(BufCal_jerk);
            debug_printl_real1(0);
            debug_printt_real1(nowpoint_tcp[1]); debug_printt_real1(nowpoint_tcp[2]); debug_printt_real1(nowpoint_tcp[3]); debug_printt_real1(nowpoint_tcp[4]); debug_printt_real1(nowpoint_tcp[5]); debug_printl_real1(nowpoint_tcp[6]);
            debug_printt_real1(auxpoint_tcp[1]); debug_printt_real1(auxpoint_tcp[2]); debug_printt_real1(auxpoint_tcp[3]); debug_printt_real1(auxpoint_tcp[4]); debug_printt_real1(auxpoint_tcp[5]); debug_printl_real1(auxpoint_tcp[6]);
            debug_printt_real1(endpoint_tcp[1]); debug_printt_real1(endpoint_tcp[2]); debug_printt_real1(endpoint_tcp[3]); debug_printt_real1(endpoint_tcp[4]); debug_printt_real1(endpoint_tcp[5]); debug_printl_real1(endpoint_tcp[6]);
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO 6 DO
                    debug_printt_real1(pointbuf_tcp[6 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_dtime[i]);
            END_FOR;
        END_IF;
    END_FOR;

    IF NOT Error THEN
        IF AxesGroup_Interrupt = 1 THEN
            stage := 8(* FBS_Inerrupt *);
        ELSE
            stage := 7(* FBS_RunBlend *);
        END_IF;

        FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
            pp_blend_exe[i] := TRUE;
        END_FOR;

        IF debug_print_TCP THEN
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO 6 DO
                    debug_printt_real1(pointbuf_tcp[6 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_dtime[i]);
            END_FOR;
        END_IF;

        IF debug_print_ACP THEN
            debug_printl_real1(0);
            FOR i := 1 TO pointnumber DO
                FOR j := 1 TO AxesGroup_Axis_Sum_In_Group - 1 DO
                    debug_printt_real1(pointbuf_acp[8 * (i - 1) + j]);
                END_FOR;
                    debug_printl_real1(pointbuf_acp[8 * (i - 1) + j]);
            END_FOR;
        END_IF;
    ELSE
        stage := 0(* FBS_WaitExec *);
    END_IF;
END_IF;

IF stage = 6(* FBS_CalcBlend *) THEN
    AxesGroup_Buffer_Busy := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 417(* AG_BBZ *));
END_IF;

MC_PP_Blend1(Axis_num := axesgroup + 0,Execute := pp_blend_exe[1],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus1[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don1,Busy => pp_blend_bus1,Active => pp_blend_act1,CommandAborted => pp_blend_abr1,Error => pp_blend_err1,ErrorID => pp_blend_eid1) ;
MC_PP_Blend2(Axis_num := axesgroup + 1,Execute := pp_blend_exe[2],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus2[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don2,Busy => pp_blend_bus2,Active => pp_blend_act2,CommandAborted => pp_blend_abr2,Error => pp_blend_err2,ErrorID => pp_blend_eid2) ;
MC_PP_Blend3(Axis_num := axesgroup + 2,Execute := pp_blend_exe[3],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus3[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don3,Busy => pp_blend_bus3,Active => pp_blend_act3,CommandAborted => pp_blend_abr3,Error => pp_blend_err3,ErrorID => pp_blend_eid3) ;
MC_PP_Blend4(Axis_num := axesgroup + 3,Execute := pp_blend_exe[4],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus4[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don4,Busy => pp_blend_bus4,Active => pp_blend_act4,CommandAborted => pp_blend_abr4,Error => pp_blend_err4,ErrorID => pp_blend_eid4) ;
MC_PP_Blend5(Axis_num := axesgroup + 4,Execute := pp_blend_exe[5],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus5[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don5,Busy => pp_blend_bus5,Active => pp_blend_act5,CommandAborted => pp_blend_abr5,Error => pp_blend_err5,ErrorID => pp_blend_eid5) ;
MC_PP_Blend6(Axis_num := axesgroup + 5,Execute := pp_blend_exe[6],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus6[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don6,Busy => pp_blend_bus6,Active => pp_blend_act6,CommandAborted => pp_blend_abr6,Error => pp_blend_err6,ErrorID => pp_blend_eid6) ;
MC_PP_Blend7(Axis_num := axesgroup + 6,Execute := pp_blend_exe[7],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus7[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don7,Busy => pp_blend_bus7,Active => pp_blend_act7,CommandAborted => pp_blend_abr7,Error => pp_blend_err7,ErrorID => pp_blend_eid7) ;
MC_PP_Blend8(Axis_num := axesgroup + 7,Execute := pp_blend_exe[8],ContinuousUpdate := FALSE,TimeScale := 1,PositionScale := 1,Offset := 0,IsAbsolute := TRUE,Position := pointbuf_pus8[1] ,DeltaTime := pointbuf_dtime[1] ,NumberOfPairs := pointnumber,BufferMode := 0,Axis_numOut => ,Done => pp_blend_don8,Busy => pp_blend_bus8,Active => pp_blend_act8,CommandAborted => pp_blend_abr8,Error => pp_blend_err8,ErrorID => pp_blend_eid8) ;

pp_blend_don[1] := pp_blend_don1; pp_blend_don[2] := pp_blend_don2; pp_blend_don[3] := pp_blend_don3; pp_blend_don[4] := pp_blend_don4; pp_blend_don[5] := pp_blend_don5; pp_blend_don[6] := pp_blend_don6; pp_blend_don[7] := pp_blend_don7; pp_blend_don[8] := pp_blend_don8;
pp_blend_bus[1] := pp_blend_bus1; pp_blend_bus[2] := pp_blend_bus2; pp_blend_bus[3] := pp_blend_bus3; pp_blend_bus[4] := pp_blend_bus4; pp_blend_bus[5] := pp_blend_bus5; pp_blend_bus[6] := pp_blend_bus6; pp_blend_bus[7] := pp_blend_bus7; pp_blend_bus[8] := pp_blend_bus8;
pp_blend_act[1] := pp_blend_act1; pp_blend_act[2] := pp_blend_act2; pp_blend_act[3] := pp_blend_act3; pp_blend_act[4] := pp_blend_act4; pp_blend_act[5] := pp_blend_act5; pp_blend_act[6] := pp_blend_act6; pp_blend_act[7] := pp_blend_act7; pp_blend_act[8] := pp_blend_act8;
pp_blend_abr[1] := pp_blend_abr1; pp_blend_abr[2] := pp_blend_abr2; pp_blend_abr[3] := pp_blend_abr3; pp_blend_abr[4] := pp_blend_abr4; pp_blend_abr[5] := pp_blend_abr5; pp_blend_abr[6] := pp_blend_abr6; pp_blend_abr[7] := pp_blend_abr7; pp_blend_abr[8] := pp_blend_abr8;

pp_blend_done := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_blend_done := pp_blend_done AND pp_blend_don[i];
END_FOR;
pp_blend_busy := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_blend_busy := pp_blend_busy AND pp_blend_bus[i];
END_FOR;
pp_blend_active_last := pp_blend_active;
pp_blend_active := AxesGroup_Axis_Sum_In_Group <> 0;
FOR i := 1 TO AxesGroup_Axis_Sum_In_Group DO
    pp_blend_active := pp_blend_active AND pp_blend_act[i];
END_FOR;

IF stage = 7(* FBS_RunBlend *) THEN
    IF AxesGroup_Buffered_Transition_Mode = 0 THEN
        IF pp_blend_done THEN
            now_motion_done := TRUE;
        END_IF;
    ELSIF AxesGroup_Buffered_Transition_Mode = 1(* TM_StartV *) THEN
        AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));
        IF AxesGroup_Now_TCP_Velocity <> 0 AND AxesGroup_Now_TCP_Velocity * 1000 <= AxesGroup_Buffered_Transition_Parameter * blendVel THEN
            now_motion_done := TRUE;
        END_IF;
    ELSIF AxesGroup_Buffered_Transition_Mode = 2(* TM_ConstantV *) THEN
        AxesGroup_Now_TCP_Velocity := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 648(* AG_NTV *));
        IF AxesGroup_Now_TCP_Velocity * 1000 <= AxesGroup_Buffered_Transition_Parameter * blendVel THEN
            now_motion_done := TRUE;
        END_IF;
    ELSIF AxesGroup_Buffered_Transition_Mode = 3(* TM_CornerDis *) THEN
        FOR i :=  1 TO 3 DO
            AxesGroup_Now_TCP_Position[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
        END_FOR;
        d := ESQR((AxesGroup_Now_TCP_Position[1] - endpoint_tcp[1]) * (AxesGroup_Now_TCP_Position[1] - endpoint_tcp[1]) +
                  (AxesGroup_Now_TCP_Position[2] - endpoint_tcp[2]) * (AxesGroup_Now_TCP_Position[2] - endpoint_tcp[2]) +
                  (AxesGroup_Now_TCP_Position[3] - endpoint_tcp[3]) * (AxesGroup_Now_TCP_Position[3] - endpoint_tcp[3]));
        s_to_end := 2.0 * radius * EASIN(d / radius / 2) + path_s[i - 1];
        IF s_to_end <= AxesGroup_Buffered_Transition_Parameter THEN
            now_motion_done := TRUE;
        END_IF;
    ELSIF AxesGroup_Buffered_Transition_Mode = 4(* TM_MaxCornerDeviation *) THEN
        FOR i :=  1 TO 3 DO
            AxesGroup_Now_TCP_Position[i] := MC_RdAxesPar_FL(AxesNum := axesgroup,AddrOffset := 316(* AG_NTP *) + 4 * (i - 1));
        END_FOR;
        d := ESQR((AxesGroup_Now_TCP_Position[1] - endpoint_tcp[1]) * (AxesGroup_Now_TCP_Position[1] - endpoint_tcp[1]) +
                  (AxesGroup_Now_TCP_Position[2] - endpoint_tcp[2]) * (AxesGroup_Now_TCP_Position[2] - endpoint_tcp[2]) +
                  (AxesGroup_Now_TCP_Position[3] - endpoint_tcp[3]) * (AxesGroup_Now_TCP_Position[3] - endpoint_tcp[3]));
        s_to_end := 2.0 * radius * EASIN(d / radius / 2) + path_s[i - 1];
        IF s_to_end <= AxesGroup_Buffered_Transition_Parameter THEN
            now_motion_done := TRUE;
        END_IF;
    END_IF;

    IF now_motion_done THEN
        now_motion_done := FALSE;
        Busy := FALSE;
        Active := FALSE;
        Done := NOT CommandAborted;

        stage := 0(* FBS_WaitExec *);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *), Data := 0);
    END_IF;
END_IF;

IF stage = 8(* FBS_Inerrupt *) THEN
    AxesGroup_Interrupt := MC_RdAxesPar_UB(AxesNum := axesgroup,AddrOffset := 422(* AG_TINR *));
    IF pp_blend_done THEN
        Active := FALSE;
        FOR i := 1 TO 6 DO
            pp_exe[i] := FALSE;
            pp_blend_exe[i] := FALSE;
        END_FOR;
    ELSIF AxesGroup_Interrupt = 0 AND NOT Active THEN
        BufCal_dec := dec;
        BufCal_jerk := jerk;
        stage := 3(* FBS_CalcRoute *);
    END_IF;
END_IF;

IF Error AND Active THEN
    IF pp_done OR pp_blend_done THEN
        Busy := FALSE;
        Active := FALSE;

        stage := 0(* FBS_WaitExec *);

        WRParRes := MC_WrAxesPar_UB(AxesNum := axesgroup,AddrOffset := 416(* AG_BZ *), Data := 0);
    END_IF;
END_IF;
lastExe := Execute;

    IF debug = 1 THEN
        AxesGroup := 7; CircMode := 1(* CM_Border *); PathChoice := 1(* PC_Clock *); CoordSystem := 2(* CS_MCS *);
        Velocity := 60; Acceleration := 20; Deceleration := 30; Jerk := 10;
        AuxPoint[1] := 200; AuxPoint[2] := 200; AuxPoint[3] := 0; AuxPoint[4] := 0; AuxPoint[5] := 0; AuxPoint[6] := 0;
        EndPoint[1] := 400; EndPoint[2] := 000; EndPoint[3] := 0; EndPoint[4] := 0; EndPoint[5] := 0; EndPoint[6] := 0;
        BufferMode := 0; TransitionMode := 0; TransitionParameter := 0;
        debug_print_PATH := TRUE; debug_print_TCP := TRUE;
    ELSIF debug = 2 THEN
        AxesGroup := 7; CircMode := 1(* CM_Border *); PathChoice := 1(* PC_Clock *); CoordSystem := 2(* CS_MCS *);
        Velocity := 60; Acceleration := 20; Deceleration := 30; Jerk := 0;
        AuxPoint[1] := 200; AuxPoint[2] := 200; AuxPoint[3] := 0; AuxPoint[4] := 0; AuxPoint[5] := 0; AuxPoint[6] := 0;
        EndPoint[1] := 400; EndPoint[2] := 000; EndPoint[3] := 0; EndPoint[4] := 0; EndPoint[5] := 0; EndPoint[6] := 0;
        BufferMode := 5; TransitionMode := 3; TransitionParameter := 100;
        debug_print_PATH := TRUE; debug_print_TCP := TRUE;
    END_IF;

END_PROGRAM