PROGRAM TEST19
VAR
    CircMode : USINT;
    TransitionMode : USINT;
    Active : BOOL;
    Busy : BOOL;
    BufferMode : USINT;
    ErrorID : UINT;
    AxesGroupOut : USINT;
    AuxPoint : ARRAY[1..8] OF REAL;
    Error : BOOL;
    PathChoice : USINT;
    CoordSystem : USINT;
    AxesGroup : USINT;
    Execute : BOOL;
    Velocity : REAL;
    Acceleration : REAL;
    Jerk : REAL;
    EndPoint : ARRAY[1..8] OF REAL;
    TransitionParameter : REAL;
    Deceleration : REAL;
END_VAR

IF stage = 0(* FBS_WaitExec *) AND Execute AND NOT lastExe AND NOT Active THEN
    IF AxesGroup &lt; 0(* AXESGROUP_MINNUM *) OR AxesGroup &gt; 20(* AXESGROUP_MAXNUM *) THEN
        ErrorID := 16#05(* ER_AxesGroupNum_ParameterError *);
    END_IF;
    AxesGroup_State := MC_RdAxesPar_UB(AxesNum := AxesGroup,AddrOffset := 4(* AG_GS *));
    IF AxesGroup_State = 0(* GS_GroupDisabled *) OR AxesGroup_State = 1(* GS_GroupErrorstop *) THEN
        ErrorID := 16#01 (* ER_StateError *);
    ELSIF AxesGroup_State = 4(* GS_GroupStopping *) THEN
        ErrorID := 16#20(* ER_AxisState_Stopping_Error *);
    END_IF;
    IF Velocity &lt; 0 OR Acceleration &lt;= 0 OR Deceleration &lt;= 0 OR Jerk &lt; 0 THEN
        ErrorID := 16#02(* ER_ParameterError *);
    END_IF;
    IF CoordSystem &lt;&gt; 1(* CS_ACS *) AND CoordSystem &lt;&gt; 2(* CS_MCS *) AND CoordSystem &lt;&gt; 3(* CS_PCS *) THEN
        ErrorID := 16#06(* ER_CoordSystem_ParameterError *);
    END_IF;
    IF BufferMode &gt;= 6 THEN
        ErrorID := 16#02(* ER_ParameterError *);
    END_IF;
    IF TransitionMode &gt;= 5 THEN
        ErrorID := 16#02(* ER_ParameterError *);
    END_IF;
    
    IF ErrorID &lt;&gt; 0(* ER_NoError *) THEN
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
END_PROGRAM