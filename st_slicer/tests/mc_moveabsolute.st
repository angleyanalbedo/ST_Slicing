PROGRAM MC_MoveAbsolute_Init_Acc

VAR
    FB_S_Type_Motion_Par_Calc_3_Segments1 : FB_S_Type_Motion_Par_Calc_3_Segments;
    FB_S_Type_Motion_Par_Calc_Fix_S1 : FB_S_Type_Motion_Par_Calc_Fix_S;

    Axis_state : USINT;       // 56 轴状态
    Axis_maxvel : REAL;       // 36 轴允许的最大速度
    Axis_maxacc : REAL;       // 40
    Axis_maxdec : REAL;       // 44
    Axis_maxjerk : REAL;      // 48
    Axis_run_number : USINT;  // 64
    Axis_mode : UINT;         // 68
    Axis_taget_postion : REAL;// 92
    Axis_relative_postion : REAL;    // 96 相对偏移
    Axis_now_velocity : REAL;        // 104 当前速度,ms级,值带方向
    Axis_blending_v :REAL;           // 132 目标混成速度
    Axis_now_motion_vel : REAL;      // 140 当前功能块目标速度,ms级
    Axis_now_motion_aimpos : REAL;        // 144 当前功能块目标位置
    Axis_run_next : USINT;                // 148
    Axis_buffered_without_blending :USINT;// 152
    Axis_busy : USINT;                    // 153
    Axis_buffer_busy : USINT;             // 155
    Axis_now_motion_direction : INT;      // 156 当前功能块运动方向，为什么要用INT型？
    Axis_last_target_postion : REAL;      // 176

    rt1 : REAL;
    rt2 : REAL;
    rt3 : REAL;
    t1 : REAL;
    t2 : REAL;
    t3 : REAL;
    t4 : REAL;
    t5 : REAL;
    t6 : REAL;
    t7 : REAL;
    t8 : REAL;
    t9 : REAL;
    t_now : UDINT;            // moving阶段刚开始时为1，每个周期（1ms）+1，结束recerse后会重置
    t_all : REAL;

    WRAxisPar_Res : BOOL;     // 写轴参数返回值
    stage : USINT;            // 功能块状态机 0：等待激活运行 1：等待Buffer 2:在Buffer中等待前一运动完成 3：计算轨迹 4：运动中
    execute_before : BOOL;    // 用于上升沿判断
    run_numberF : USINT;      // 此次运动的运行ID，用来判断有没有被打断

    pos : REAL;               // 此次运动目标位置
    start_postion : REAL;     // 此次运动的起始位置
    direction : INT;          // 此次运动的运动方向（1：正方向，-1：负方向），类型与轴全局变量一致
    jerk_mode : BOOL;         // 此次运动的加速度形式 TRUE:S型，FALSE:梯形
    run_mode : USINT;         // 此次运动的速度形式

    vel : REAL;
    acc : REAL;
    dec : REAL;
    jerk : REAL;
    rvs : REAL;               // 此次运动反方向初速度
    vs : REAL;                // 此次运动初速度，反向减速后置为0，非负
    vmax : REAL;              // 此次运动 可达到 的最大速度，正
    now_direction : INT;      // 当前运动方向（1/-1）
    now_stage_vs : REAL;      // 当前阶段初速度
    blend_start_a : REAL;     // 混成阶段开始时的加速度
    blend_start_t : REAL;     // 混成阶段开始时的时刻
    blend_start_v : REAL;     // 混成阶段开始时的速度
    blend_start_pos : REAL;   // 混成阶段开始时的绝对位置
    s_blend : REAL;           // 混成阶段剩余的距离
    blend_equal_vel : REAL;   // 混成匀速阶段速度

    rs1 : REAL;
    rs2 : REAL;
    rs3 : REAL;
    s1 : REAL;
    s2 : REAL;
    s3 : REAL;
    s4 : REAL;
    s5 : REAL;
    s6 : REAL;
    s7 : REAL;
    s8 : REAL;
    s9 : REAL;
    s_rev : REAL;                // S型运动三段减速距离和
    s_all : REAL;                // 此次运动设定的路程，非负，如果需要先反向减速（direction &lt;&gt; now_direction）则会再加上减速距离
    s_acc : REAL;                // S型运动：加速阶段距离
    s_dec : REAL;                // S型运动：减速阶段距离
    blend_s1 : REAL;
    blend_s2 : REAL;

    debug_print : BOOL;
    debug_print_count : USINT;
    nouse_ub : USINT;
END_VAR

VAR_INPUT
    Axis_num :USINT;
    Execute : BOOL;
    Position : REAL;
    Velocity : REAL;
    Acceleration : REAL;
    Deceleration : REAL;
    Jerk : REAL;
    BufferMode : USINT;
END_VAR

VAR_OUTPUT
    Axis_numOut : USINT;
    Done : BOOL;
    Busy : BOOL;
    Active : BOOL;
    CommandAborted : BOOL;
    Error : BOOL;
    ErrorID : UINT;
END_VAR

IF NOT Execute AND stage &lt;&gt; 3(* FBS_InBuffer *) AND stage &lt;&gt; 4(* FBS_Moving *) THEN
    Axis_numOut := Axis_num;
    Done := FALSE;
    Active := FALSE;
    Busy := FALSE;
    CommandAborted := FALSE;
    Error := FALSE;
    ErrorID := 0(* ER_NoError *);

    stage := 0(* FBS_WaitExec *);
    execute_before := FALSE;
    s_blend := 0;
END_IF;

Axis_state := MC_RdAxisPar_UB(AxisNum := Axis_num,AddrOffset := 56) ;
IF (Axis_state = 0(* AS_Disabled *) OR Axis_state = 1(* AS_Errorstop *)) AND stage = 4(* FBS_Moving *) THEN
    CommandAborted := TRUE;
    Error := TRUE;
    ErrorID := 16#01(* ER_StateError *);
    stage := 0(* FBS_WaitExec *);
END_IF;

IF stage = 0(* FBS_WaitExec *) AND Execute AND NOT execute_before THEN
    Axis_maxvel := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 36); 
    Axis_maxacc := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 40); 
    Axis_maxdec := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 44); 
    Axis_maxjerk := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 48); 
    IF Axis_num &gt;= 50(* AXIS_MAXNUM *) OR Velocity &lt;= 0.004 (* MINV *) OR Velocity &gt; Axis_maxvel OR
       Acceleration &lt;= 0.004 (* MINV *) OR Acceleration &gt; Axis_maxacc OR
       Deceleration &lt;= 0.004 (* MINV *) OR Deceleration &gt; Axis_maxdec OR
       Jerk &lt; 0 OR Jerk &gt; Axis_maxjerk 
    THEN
        Error := TRUE;
        ErrorID := 16#02(* ER_ParameterError *);

        debug_ub := F_debug_print_ub(26); debug_ub := F_debug_printl_ub(1);   // 26_1
        debug_ub := F_debug_printl_real(Axis_num);
        debug_ub := F_debug_print_real(Velocity); debug_ub := F_debug_printl_real(Axis_maxvel);
        debug_ub := F_debug_print_real(Acceleration); debug_ub := F_debug_printl_real(Axis_maxacc);
        debug_ub := F_debug_print_real(Deceleration); debug_ub := F_debug_printl_real(Axis_maxdec);
        debug_ub := F_debug_print_real(Jerk); debug_ub := F_debug_printl_real(Axis_maxjerk);
    ELSIF Axis_state = 0(* MC_Disabled *) OR Axis_state = 1(* MC_Errorstop *) OR Axis_state = 4(* MC_Stopping *) THEN
        Error := TRUE;
        ErrorID := 16#01(* ER_StateError *);
    ELSE
        Error := FALSE;
        ErrorID := 16#00(* ER_NoError *);
        Axis_relative_postion := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 96) ;
        vel := Velocity / 1000.0;
        acc := Acceleration / 1000000.0;
        dec := Deceleration / 1000000.0;
        jerk := Jerk / 1000000000.0;
        pos := Position - Axis_relative_postion;
        stage := 1(* FBS_WaitBuffer *);
    END_IF;
END_IF;

IF stage = 1(* FBS_WaitBuffer *)  THEN
    IF BufferMode = 0(* BM_Aborting *) THEN
        stage := 3(* FBS_CalcRoute *);
    ELSE
        Axis_buffer_busy := MC_RdAxisPar_UB(AxisNum := Axis_num,AddrOffset := 155);
        IF Axis_buffer_busy = 0 THEN
            Axis_run_next := MC_RdAxisPar_UB(AxisNum := Axis_num,AddrOffset := 148);

            IF BufferMode = 1(* BM_Buffered *) AND Axis_run_next &lt;&gt; 1 THEN
                WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 152, Data := 1);
            END_IF;

            IF BufferMode = 2(* BM_Blendinglow *) THEN
                Axis_now_motion_vel := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 140); 
                IF Axis_now_motion_vel &gt; vel THEN
                    WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 132, Data := vel);
                ELSE
                    WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 132, Data := Axis_now_motion_vel);
                END_IF;
            END_IF;

            IF BufferMode = 3(* BFM_BlendingPrevious *) THEN
                Axis_now_motion_vel := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 140);
                WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 132, Data := Axis_now_motion_vel);
            END_IF;

            IF BufferMode = 4 (*BM_BlendingNext*) THEN
                Axis_now_motion_vel := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 140);
                WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 132, Data := vel);
            END_IF;

            IF BufferMode = 5(*BM_BlendingHigh*) THEN
                Axis_now_motion_vel := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 140); 
                IF Axis_now_motion_vel &lt; vel THEN
                    WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 132, Data := vel);
                ELSE
                    WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 132, Data := Axis_now_motion_vel);
                END_IF;
            END_IF;

            IF (BufferMode = 1 AND Axis_run_next &lt;&gt; 1) OR BufferMode &lt;&gt; 1 THEN
                WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 155, Data := 1);
                stage := 2(* FBS_InBuffer *);
            END_IF;
        END_IF;
    END_IF;
END_IF;

IF stage = 2(* FBS_InBuffer *) THEN
    Axis_busy := MC_RdAxisPar_UB(AxisNum := Axis_num,AddrOffset := 153);
    IF Axis_busy = 0 THEN
        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 155, Data := 0);
        stage := 3(* FBS_CalcRoute *);
    END_IF;
END_IF;

IF stage = 3(* FBS_CalcRoute *) THEN
    Axis_run_number := MC_RdAxisPar_UB(AxisNum := Axis_num,AddrOffset := 64);
    Axis_run_number := Axis_run_number MOD 255 + 1; // [1,255]
    WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 64, Data := Axis_run_number);
    run_numberF := Axis_run_number;

    WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 153, Data := 1);
    WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 152, Data := 0);

    start_postion := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 176);

    IF pos &gt;= start_postion THEN
        direction := 1;
        s_all := pos - start_postion;
    ELSE
        direction := -1;
        s_all := start_postion - pos;
    END_IF;

    vs := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 104);
    IF vs &lt;= -0.004(* MINV *) THEN
        vs := -1.0 * vs;
        now_direction := -1;
    ELSIF vs &gt;= 0.004(* MINV *) THEN
        now_direction := 1;
    ELSE
        vs := 0;
        now_direction := direction;
    END_IF;
    as := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 168);

    IF jerk &gt; 0 THEN
        jerk_mode := TRUE;
    ELSE
        jerk_mode := FALSE;
    END_IF;

    WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 144, Data := pos);
    WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 140, Data := vel);

    IF s_all &lt;= 0.001(* MINP *) THEN
        Done := TRUE;
        CommandAborted := FALSE;
        Busy := FALSE;
        Active := FALSE;
        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 153, Data := 0);
        stage := 0(* FBS_WaitExec *);
    ELSE
        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 56, Data := 5(* AS_Discrete_Motion *));

        IF NOT jerk_mode THEN    // 梯形模式
            IF direction &lt;&gt; now_direction THEN    // 目标方向与当前速度方向不同,先减速至0
                rt1 := vs / dec;
                rs1 := vs * vs / dec / 2;
                s_all := s_all + rs1;
                rvs := vs;
                vs := 0;
            END_IF;

            IF vs &lt; vel - 0.004(* MINV *) THEN    // 运动初速度 &lt; 运动目标速度
                run_mode := 1(* RM_ACC_EQU_DEC *);
                vmax := vel;
                t1 := (vmax - vs) / acc;
                t3 := vmax / dec;
                s1 := Motion_Delta_S(vs := vs,acc := acc,dec := 0,jerk := 0,d_t := t1);
                s3 := Motion_Delta_S(vs := vmax,acc := 0,dec := dec,jerk := 0,d_t := t3);
                s2 := s_all - s1 - s3;
                t2 := s2 / vmax;
                IF t2 &lt; 0 THEN
                    t3 := vs / dec;
                    s3 := Motion_Delta_S(vs := vs,acc := 0,dec := dec,jerk := 0,d_t := t3);
                    IF s_all &lt;= s3 THEN    // 直接减速都超距离
                        vmax := vs;
                        t1 := 0; t2 := 0;
                        s1 := 0; s2 := 0;
                        s_all := s3;
                    ELSE
                        vmax := ESQR(vs * vs - (acc * vs * vs - 2 * acc * dec * s_all) / (acc + dec));
                        t1 := (vmax - vs) / acc;
                        t2 := 0;
                        t3 := vmax / dec;
                        s1 := vs * t1 + acc * t1 * t1 / 2;
                        s2 := 0;
                        s3 := s_all - s1;
                    END_IF;
                END_IF;
                t_all := t1 + t2 + t3;
            END_IF;

            IF vs &gt;= vel - 0.004(* MINV *) THEN   // 运动初速度 &gt;= 运动目标速度
                run_mode := 2(* RM_DEC_EQU_DEC *);
                vmax := vel;
                t1 := (vs - vmax) / dec;
                t3 := vmax / dec;
                s1 := Motion_Delta_S(vs := vs,acc := 0,dec := dec,jerk := 0,d_t := t1);
                s3 := Motion_Delta_S(vs := vmax,acc := 0,dec := dec,jerk := 0,d_t := t3);
                s2 := s_all - s1 - s3;
                t2 := s2 / vmax;
                IF t2 &lt; 0 THEN
                    t2 := 0; s2 := 0;
                    s_all := s1 + s3;
                END_IF;
                t_all := t1 + t2 + t3;
            END_IF;
        ELSE                     // S 型模式
            Axis_now_acceleration := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 168);
            as := Axis_now_acceleration;

            IF direction &lt;&gt; now_direction THEN    // 目标方向与当前速度方向不同,先减速至0
                FB_S_Type_Motion_Par_Calc_3_Segments1(vs := vs,as := as,vt := 0,acc := 0,dec := dec,jerk := jerk,
                    t1 =&gt; rt1,t2 =&gt; rt2,t3 =&gt; rt3,s1 =&gt; rs1,s2 =&gt; rs2,s3 =&gt; rs3,t_3seg =&gt; ,s_3seg =&gt; s_rev);
                s_all := s_all + s_rev;
                rvs := vs;
                vs := 0;
                as := 0;
            END_IF;

            IF vs &lt; vel THEN    // 初速度 &lt; 目标速度
                run_mode := 1(* RM_ACC_EQU_DEC *);

                vmax := vel;
                FB_S_Type_Motion_Par_Calc_3_Segments1(vs := vs,as := as,vt := vmax,acc := acc,dec := 0,jerk := jerk,
                    t1 =&gt; t1,t2 =&gt; t2,t3 =&gt; t3,s1 =&gt; s1,s2 =&gt; s2,s3 =&gt; s3,t_3seg =&gt; ,s_3seg =&gt; s_acc );
                FB_S_Type_Motion_Par_Calc_3_Segments1(vs := vmax,as := 0,vt := 0,acc := 0,dec := dec,jerk := jerk,
                    t1 =&gt; t5,t2 =&gt; t6,t3 =&gt; t7,s1 =&gt; s5,s2 =&gt; s6,s3 =&gt; s7,t_3seg =&gt; ,s_3seg =&gt; s_dec );
                s4 := s_all - s_acc - s_dec;
                t4 := s4 / vmax;
                t_all := t1 + t2 + t3 + t4 + t5 + t6 + t7;

                IF t4 &lt; 0 THEN    // 达不到最大速度
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := vs,as := as,vt := 0,acc := 0,dec := dec,jerk := jerk,
                        t1 =&gt; t5,t2 =&gt; t6,t3 =&gt; t7,s1 =&gt; s5,s2 =&gt; s6,s3 =&gt; s7,t_3seg =&gt; t_all,s_3seg =&gt; s_dec );
                    IF s_dec &gt;= s_all THEN    // 直接减速都超出预定距离
                        s_all := s_dec;
                        t1 := 0; s1 := 0;
                        t2 := 0; s2 := 0;
                        t3 := 0; s3 := 0;
                        t4 := 0; s4 := 0;
                        s_acc := 0;
                        vmax := vs;
                        t_all := t1 + t2 + t3 + t4 + t5 + t6 + t7;
                    ELSE                      // 直接减不会超过限度
                        FB_S_Type_Motion_Par_Calc_Fix_S1(vs := vs,as := as,vm := vel,vt := 0,acc := acc,dec := dec,jerk := jerk,s_all := s_all,
                            vmax =&gt; vmax,t1 =&gt; t1,t2 =&gt; t2,t3 =&gt; t3,t4 =&gt; t4,t5 =&gt; t5,t6 =&gt; t6,t7 =&gt; t7,
                            s1 =&gt; s1,s2 =&gt; s2,s3 =&gt; s3,s4 =&gt; s4,s5 =&gt; s5,s6 =&gt; s6,s7 =&gt; s7,t_all =&gt; t_all); 
                    END_IF;
                END_IF;
            END_IF;

            IF vs &gt;= vel THEN       // 初速度 &gt;= 目标速度
                run_mode := 2(*RM_DEC_EQU_DEC*);

                vmax := vel;
                FB_S_Type_Motion_Par_Calc_3_Segments1(vs := vs,as := as,vt := vmax,acc := 0,dec := dec,jerk := jerk,
                    t1 =&gt; t1,t2 =&gt; t2,t3 =&gt; t3,s1 =&gt; s1,s2 =&gt; s2,s3 =&gt; s3,t_3seg =&gt; ,s_3seg =&gt; s_acc);
                FB_S_Type_Motion_Par_Calc_3_Segments1(vs := vmax,as := 0,vt := 0,acc := 0,dec := dec,jerk := jerk,
                    t1 =&gt; t5,t2 =&gt; t6,t3 =&gt; t7,s1 =&gt; s5,s2 =&gt; s6,s3 =&gt; s7,t_3seg =&gt; ,s_3seg =&gt; s_dec);
                s4 := s_all - s_acc - s_dec;
                t4 := s4 / vmax;
                t_all := t1 + t2 + t3 + t4 + t5 + t6 + t7;

                IF t4 &lt; 0 THEN
                    FB_S_Type_Motion_Par_Calc_3_Segments1(vs := vs,as := as,vt := 0,acc := 0,dec := dec,jerk := jerk,
                        t1 =&gt; t5,t2 =&gt; t6,t3 =&gt; t7,s1 =&gt; s5,s2 =&gt; s6,s3 =&gt; s7,t_3seg =&gt; ,s_3seg =&gt; s_dec);
                    IF s_dec &gt;= s_all THEN
                        s_all := s_dec;
                        t1 := 0; s1 := 0;
                        t2 := 0; s2 := 0;
                        t3 := 0; s3 := 0;
                        t4 := 0; s4 := 0;
                        s_acc := 0;
                        vmax := vs;
                        t_all := t1 + t2 + t3 + t4 + t5 + t6 + t7;
                    ELSE
                        FB_S_Type_Motion_Par_Calc_Fix_S1(vs := vs,as := as,vm := vel,vt := 0,acc := 0,dec := dec,jerk := jerk,s_all := s_all,
                            vmax =&gt; vmax,t1 =&gt; t1,t2 =&gt; t2,t3 =&gt; t3,t4 =&gt; t4,t5 =&gt; t5,t6 =&gt; t6,t7 =&gt; t7,
                            s1 =&gt; s1,s2 =&gt; s2,s3 =&gt; s3,s4 =&gt; s4,s5 =&gt; s5,s6 =&gt; s6,s7 =&gt; s7,t_all =&gt; t_all); 
                    END_IF;
                END_IF;
            END_IF;
            as := RealAbs(as);
        END_IF;

        t_now := 1;
        stage := 4(* FBS_Moving *);

        debug_ub := F_debug_print_ub(26); debug_ub := F_debug_printl_ub(2);   // 26_2
        debug_ub := F_debug_print_ub(Axis_num); debug_ub := F_debug_printl_real(as);
        debug_ub := F_debug_print_real(t1); debug_ub := F_debug_print_real(t2); debug_ub := F_debug_print_real(t3); 
        debug_ub := F_debug_print_real(t4); debug_ub := F_debug_print_real(t5); debug_ub := F_debug_print_real(t6); debug_ub := F_debug_printl_real(t7);
    END_IF;

END_IF;

IF stage = 4(* FBS_Moving *) THEN
    Axis_run_number := MC_RdAxisPar_UB(AxisNum := Axis_num,AddrOffset := 64);
    IF Axis_run_number &lt;&gt; run_numberF THEN    // 功能块被打断
        CommandAborted := TRUE;
        run_numberF := 0;
        Busy := FALSE;
        Active := FALSE;
        Done := FALSE;
        stage := 0(* FBS_WaitExec *);
    ELSE
        Busy := TRUE;
        Active := TRUE;
        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 153, Data := 1);
        IF NOT jerk_mode THEN    // 梯形模式
            IF direction &lt;&gt; now_direction THEN
                IF t_now &gt;= rt1 THEN
                    Axis_taget_postion := start_postion + now_direction * rs1;
                    start_postion := start_postion + now_direction * rs1;
                    now_direction := direction;
                    t_now := 0;
                ELSE
                    Axis_taget_postion := start_postion + now_direction *
                        Motion_Delta_S(vs := rvs,acc := 0,dec := dec,jerk := 0,d_t := t_now); 
                END_IF;
            ELSE
                IF run_mode = 1(*RM_ACC_EQU_DEC*) OR run_mode = 2(*RM_DEC_EQU_DEC*) THEN
                    IF t_now &gt;= t_all THEN
                        Busy := FALSE;
                        Active := FALSE;
                        Done := TRUE;
                        CommandAborted := FALSE;

                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 148, Data := 1);
                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 153, Data := 0);

                        stage := 0;

                        Axis_taget_postion := start_postion + now_direction * s_all;
                    ELSIF t_now &lt; t1 THEN
                        IF run_mode = 1(* RM_ACC_EQU_DEC *) THEN
                            Axis_taget_postion := start_postion + now_direction * 
                                Motion_Delta_S(vs := vs,acc := acc,dec := 0,jerk := 0,d_t := t_now); 
                        ELSE
                            Axis_taget_postion := start_postion + now_direction * 
                                Motion_Delta_S(vs := vs,acc := 0,dec := dec,jerk := 0,d_t := t_now);
                        END_IF;
                    ELSIF t_now &lt; t1 + t2 THEN
                        Axis_taget_postion := start_postion + now_direction * ( s1 + 
                            Motion_Delta_S(vs := vmax,acc := 0,dec := 0,jerk := 0,d_t := t_now - t1));
                    ELSIF t_now &lt; t_all THEN
                        Axis_taget_postion := start_postion + now_direction * ( s1 + s2 +
                            Motion_Delta_S(vs := vmax,acc := 0,dec := dec,jerk := 0,d_t := t_now - t1 - t2));
                    END_IF;
                ELSIF run_mode = 3(* RM_BLENDING_EQU_DEC *) OR run_mode = 4(* RM_BLENDING_EQU_DEC *) THEN
                    IF t_now &gt;= t_all THEN
                        Busy := FALSE;
                        Active := FALSE;
                        Done := TRUE;
                        CommandAborted := FALSE;
                        stage := 0;

                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 148, Data := 1);
                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 153, Data := 0);

                        Axis_blending_v := MC_RdAxisPar_FL(AxisNum  := Axis_num, AddrOffset := 132);
                        Axis_taget_postion := MC_RdAxisPar_FL(AxisNum  := Axis_num, AddrOffset := 92);
                        Axis_taget_postion := Axis_taget_postion + direction * Axis_blending_v;
                    ELSIF t_now &lt;= blend_start_t + t4 THEN
                        Axis_taget_postion := blend_start_pos + now_direction * 
                            Motion_Delta_S(vs := blend_equal_vel,acc := 0,dec := 0,jerk := 0,d_t := (t_now - blend_start_t));
                    ELSE
                        IF run_mode = 3(* RM_BLENDING_EQU_DEC *) THEN
                            Axis_taget_postion := blend_start_pos + now_direction * ( s4 +
                                Motion_Delta_S(vs := blend_equal_vel,acc := 0,dec := dec,jerk := 0,d_t := (t_now - blend_start_t - t4)));
                        ELSE
                            Axis_taget_postion := blend_start_pos + now_direction * ( s4 +
                                Motion_Delta_S(vs := blend_equal_vel,acc := acc,dec := 0,jerk := 0,d_t := (t_now - blend_start_t - t4)));
                        END_IF;
                    END_IF;
                END_IF;
            END_IF;
        ELSE                     // S 型模式
            IF direction &lt;&gt; now_direction THEN // 先减速至0
                IF t_now &gt;= rt1 + rt2 +rt3 THEN
                    Axis_taget_postion := start_postion + now_direction * s_rev;
                    start_postion := start_postion + now_direction * s_rev;
                    now_direction := direction;
                    t_now := 0;
                ELSIF t_now &lt;= rt1 THEN
                    now_stage_vs := rvs;
                    Axis_taget_postion := start_postion + now_direction *
                        Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := 0,jerk := ( -1 * jerk),d_t := t_now);
                ELSIF t_now &lt;= rt1 + rt2 THEN
                    now_stage_vs := rvs - jerk * rt1 * rt1 / 2;
                    Axis_taget_postion := start_postion + now_direction * ( rs1 +
                        Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := dec,jerk := 0,d_t := t_now - rt1));
                ELSIF t_now &lt; rt1 + rt2 + rt3 THEN
                    now_stage_vs := rvs - jerk * rt1 * rt1 / 2 - jerk * rt1 * rt2;
                    Axis_taget_postion := start_postion + now_direction * ( rs1 + rs2 + 
                        Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := jerk * rt1,jerk := jerk,d_t := t_now - rt1 - rt2));
                END_IF;
            ELSE
                IF run_mode = 1(* RM_ACC_EQU_DEC *) OR run_mode = 2(* RM_DEC_EQU_DEC *) THEN
                    IF t_now &gt;= t_all THEN
                        Busy := FALSE;
                        Active := FALSE;
                        Done := TRUE;
                        CommandAborted := FALSE;

                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 148, Data := 1);
                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 153, Data := 0);

                        stage := 0;

                        Axis_taget_postion := start_postion + now_direction * s_all;
                    ELSIF t_now &lt;= t1 THEN
                        now_stage_vs := vs;
                        IF run_mode = 1(* RM_ACC_EQU_DEC *) THEN
                            Axis_taget_postion := start_postion + now_direction * 
                                Motion_Delta_S(vs := now_stage_vs,acc := as,dec := 0,jerk := jerk,d_t := t_now); 
                        ELSE
                            Axis_taget_postion := start_postion + now_direction * 
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := as,jerk := ( -1 * jerk),d_t := t_now);
                        END_IF;
                    ELSIF t_now &lt;= t1 + t2 THEN
                        IF run_mode = 1(* RM_ACC_EQU_DEC *) THEN
                            now_stage_vs := vs + jerk * t1 * t1 / 2;
                            Axis_taget_postion := start_postion + now_direction * ( s1 +
                                Motion_Delta_S(vs := now_stage_vs,acc := acc,dec := 0,jerk := 0,d_t := t_now - t1));    // 这里可以填acc是因为到达匀加速阶段时的加速度一定达到预设值
                        ELSE
                            now_stage_vs := vs - jerk * t1 * t1 / 2;
                            Axis_taget_postion := start_postion + now_direction * ( s1 +
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := dec,jerk := 0,d_t := t_now - t1));
                        END_IF;
                    ELSIF t_now &lt;= t1 + t2 + t3 THEN
                        IF run_mode = 1(* RM_ACC_EQU_DEC *) THEN
                            now_stage_vs := vs + jerk * t1 * t1 / 2 + jerk * t1 * t2;
                            Axis_taget_postion := start_postion + now_direction * ( s1 + s2 + 
                                Motion_Delta_S(vs := now_stage_vs,acc := jerk * t1,dec := 0,jerk := ( -1 * jerk),d_t := t_now - t1 - t2));
                        ELSE
                            now_stage_vs := vs - jerk * t1 * t1 / 2 - jerk * t1 * t2;
                            Axis_taget_postion := start_postion + now_direction * ( s1 + s2 + 
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := jerk * t1,jerk := jerk,d_t := t_now - t1 - t2));
                        END_IF;
                    ELSIF t_now &lt;= t1 + t2 + t3 + t4 THEN
                        now_stage_vs := vmax;
                        Axis_taget_postion := start_postion + now_direction * ( s1 + s2 + s3 + 
                            Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := 0,jerk := 0,d_t := t_now - t1 - t2 - t3));
                    ELSIF t_now &lt;= t1 + t2 + t3 + t4 + t5 THEN
                        now_stage_vs := vmax;
                        Axis_taget_postion := start_postion + now_direction * ( s1 + s2 + s3 + s4 +
                            Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := 0,jerk := ( -1 * jerk),d_t := t_now - t1 - t2 - t3 - t4));
                    ELSIF t_now &lt;= t1 + t2 + t3 + t4 + t5 + t6 THEN
                        now_stage_vs := vmax - jerk * t5 * t5 / 2;
                        Axis_taget_postion := start_postion + now_direction * ( s1 + s2 + s3 + s4 + s5 +
                            Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := jerk * t5,jerk := 0,d_t := t_now - t1 - t2 - t3 - t4 - t5));
                    ELSIF t_now &lt;= t1 + t2 + t3 + t4 + t5 + t6 + t7 THEN
                        now_stage_vs := vmax - jerk * t5 * t5 / 2 - jerk * t5 * t6;
                        Axis_taget_postion := start_postion + now_direction * ( s1 + s2 + s3 + s4 + s5 + s6 + 
                            Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := jerk * t5,jerk := jerk,d_t := UDINT_TO_REAL(t_now) - t1 - t2 - t3 - t4 - t5 - t6));
                    END_IF;
                ELSIF run_mode = 3(* RM_BLENDING_EQU_DEC *) THEN
                    IF t_now &gt;= t_all THEN
                        Busy := FALSE;
                        Active := FALSE;
                        Done := TRUE;
                        CommandAborted := FALSE;
                        stage := 0;

                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 148, Data := 1);
                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 153, Data := 0);

                        Axis_blending_v := MC_RdAxisPar_FL(AxisNum  := Axis_num, AddrOffset := 132);
                        Axis_taget_postion := MC_RdAxisPar_FL(AxisNum  := Axis_num, AddrOffset := 92);
                        Axis_taget_postion := Axis_taget_postion + direction * Axis_blending_v;
                    ELSIF t_now &lt;= blend_start_t + t5 THEN
                        now_stage_vs := blend_start_v;
                        Axis_taget_postion := blend_start_pos + now_direction * 
                            Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := blend_start_a,jerk := jerk,d_t := UDINT_TO_REAL(t_now) - blend_start_t);
                    ELSIF t_now &lt;= blend_start_t + t5 + t6 THEN
                        now_stage_vs := blend_equal_vel;
                        Axis_taget_postion := blend_start_pos + now_direction * ( blend_s1 +
                            Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := 0,jerk := 0,d_t := t_now - blend_start_t - t5));
                    ELSIF t_now &lt;= blend_start_t + t5 + t6 + t7 THEN
                        now_stage_vs := blend_equal_vel; 
                        Axis_taget_postion := blend_start_pos + now_direction * ( blend_s1 + s6 +
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := 0,jerk := ( -1 * jerk),d_t := t_now - blend_start_t - t5 - t6)); 
                    ELSIF t_now &lt;= blend_start_t + t5 + t6 + t7 + t8 THEN
                        now_stage_vs := blend_equal_vel - jerk * t7 * t7 / 2; 
                        Axis_taget_postion := blend_start_pos + now_direction * ( blend_s1 + s6 + s7 +
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := dec,jerk := 0,d_t := t_now - blend_start_t - t5 - t6 - t7));
                    ELSIF t_now &lt;= blend_start_t + t5 + t6 + t7 + t8 + t9 THEN
                        now_stage_vs := blend_equal_vel - jerk * t7 * t7 / 2 - jerk * t7 * t8;
                        Axis_taget_postion := blend_start_pos + now_direction * ( blend_s1 + s6 + s7 + s8 +
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := jerk * t7,jerk := jerk,d_t := t_now - blend_start_t - t5 - t6 - t7 - t8));
                    END_IF;
                ELSIF run_mode = 4(* RM_BLENDING_EQU_ACC *) THEN
                    IF t_now &gt;= t_all THEN
                        Busy := FALSE;
                        Active := FALSE;
                        Done := TRUE;
                        CommandAborted := FALSE;
                        stage := 0;

                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 148, Data := 1);
                        WRAxisPar_Res := MC_WrAxisPar_UB(AxisNum := Axis_num, AddrOffset := 153, Data := 0);

                        Axis_blending_v := MC_RdAxisPar_FL(AxisNum  := Axis_num, AddrOffset := 132);
                        Axis_taget_postion := MC_RdAxisPar_FL(AxisNum  := Axis_num, AddrOffset := 92);
                        Axis_taget_postion := Axis_taget_postion + direction * Axis_blending_v;
                    ELSIF t_now &lt;= blend_start_t + t5 THEN
                        now_stage_vs := blend_start_v;
                        Axis_taget_postion := blend_start_pos + now_direction * 
                            Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := blend_start_a,jerk := jerk,d_t := UDINT_TO_REAL(t_now) - blend_start_t);
                    ELSIF t_now &lt;= blend_start_t + t5 + t6 THEN
                        now_stage_vs := blend_equal_vel;
                        Axis_taget_postion := blend_start_pos + now_direction * ( blend_s1 +
                            Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := 0,jerk := 0,d_t := t_now - blend_start_t - t5));
                    ELSIF t_now &lt;= blend_start_t + t5 + t6 + t7 THEN
                        now_stage_vs := blend_equal_vel; 
                        Axis_taget_postion := blend_start_pos + now_direction * ( blend_s1 + s6 +
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := 0,jerk := jerk,d_t := t_now - blend_start_t - t5 - t6)); 
                    ELSIF t_now &lt;= blend_start_t + t5 + t6 + t7 + t8 THEN
                        now_stage_vs := blend_equal_vel + jerk * t7 * t7 / 2; 
                        Axis_taget_postion := blend_start_pos + now_direction * ( blend_s1 + s6 + s7 +
                                Motion_Delta_S(vs := now_stage_vs,acc := acc,dec := 0,jerk := 0,d_t := t_now - blend_start_t - t5 - t6 - t7));
                    ELSIF t_now &lt;= blend_start_t + t5 + t6 + t7 + t8 + t9 THEN
                        now_stage_vs := blend_equal_vel + jerk * t7 * t7 / 2 + jerk * t7 * t8;
                        Axis_taget_postion := blend_start_pos + now_direction * ( blend_s1 + s6 + s7 + s8 +
                                Motion_Delta_S(vs := now_stage_vs,acc := jerk * t7,dec := 0,jerk := ( -1 * jerk),d_t := t_now - blend_start_t - t5 - t6 - t7 - t8));
                    END_IF;
                END_IF;
            END_IF;
        END_IF;

        Axis_buffer_busy := MC_RdAxisPar_UB(AxisNum := Axis_num,AddrOffset := 155);
        IF Axis_buffer_busy = 1 AND s_blend = 0 THEN    // 缓存中有内容需要进行混成计算
            Axis_buffered_without_blending := MC_RdAxisPar_UB(AxisNum := Axis_num,AddrOffset := 152);
            IF Axis_buffered_without_blending = 0 THEN
                IF (NOT jerk_mode AND t_now &gt; t1 AND t_now &lt; t_all) OR
                   (jerk_mode AND t_now &gt; t1 + t2 + t3 AND t_now &lt; t_all) THEN
                    IF jerk_mode THEN    // S型混成
                        IF t_now &lt;= t1 + t2 + t3 + t4 THEN
                            blend_start_a := 0;
                            s_blend := s_all - s1 - s2 - s3 - 
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := 0,jerk := 0,d_t := t_now - t1 - t2 - t3);
                        ELSIF t_now &lt;= t1 + t2 + t3 + t4 + t5 THEN
                            blend_start_a := jerk * (t_now - t1 - t2 - t3 - t4);
                            s_blend := s_all - s1 - s2 - s3 - s4 - 
                                Motion_Delta_S(vs := vmax,acc := 0,dec := 0,jerk := ( -1 * jerk),d_t := t_now - t1 - t2 - t3 - t4);
                        ELSIF t_now &lt;= t1 + t2 + t3 + t4 + t5 + t6 THEN
                            blend_start_a := jerk * t5;
                            s_blend := s_all - s1 - s2 - s3 - s4 - s5 - // 这里 now_stage_vs 没有发生变化
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := jerk * t5,jerk := 0,d_t := t_now - t1 - t2 - t3 - t4 - t5);
                        ELSE
                            blend_start_a := jerk * (t5 - (t_now - t1 - t2 - t3 - t4 - t5 - t6)); // 跟c代码不同，多t6
                            s_blend := s_all - s1 - s2 - s3 - s4 - s5 - s6 - 
                                Motion_Delta_S(vs := now_stage_vs,acc := 0,dec := jerk * t5,jerk := jerk,d_t := UDINT_TO_REAL(t_now) - t1 - t2 - t3 - t4 - t5 - t6);
                        END_IF;

                        IF blend_start_a &lt;&gt; 0 THEN
                            t5 := blend_start_a / jerk;
                        ELSE
                            t5 := 0;
                        END_IF;

                        Axis_blending_v := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 132);
                        Axis_now_velocity := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 104);
                        blend_start_t := t_now;
                        blend_start_v := RealAbs(Axis_now_velocity);
                        blend_start_pos := start_postion + s_all - s_blend;
                        blend_equal_vel := blend_start_v - blend_start_a * t5 + jerk * t5 * t5 / 2;

                        blend_s1 := Motion_Delta_S(vs := blend_start_v,acc := 0,dec := blend_start_a,jerk := jerk,d_t := t5);

                        IF Axis_blending_v &gt; blend_equal_vel - 0.004(* MINV *) THEN  // 混承速度 &gt; 当前速度
                            run_mode := 4(* RM_BLENDING_EQU_ACC *);
                            FB_S_Type_Motion_Par_Calc_3_Segments1(vs := blend_equal_vel,vt := Axis_blending_v,acc := acc,dec := 0,jerk := jerk,
                                t1 =&gt; t7,t2 =&gt; t8,t3 =&gt; t9,s1 =&gt; s7,s2 =&gt; s8,s3 =&gt; s9,t_3seg =&gt; ,s_3seg =&gt; blend_s2);
                        ELSE                                                       // 混承速度 &lt;= 当前速度
                            run_mode := 3(* RM_BLENDING_EQU_DEC *);
                            FB_S_Type_Motion_Par_Calc_3_Segments1(vs := blend_equal_vel,vt := Axis_blending_v,acc := 0,dec := dec,jerk := jerk,
                                t1 =&gt; t7,t2 =&gt; t8,t3 =&gt; t9,s1 =&gt; s7,s2 =&gt; s8,s3 =&gt; s9,t_3seg =&gt; ,s_3seg =&gt; blend_s2);
                        END_IF;

                        s6 := s_blend - blend_s1 - blend_s2;
                        t6 := s6 / blend_equal_vel;
                        IF t6 &lt; 0 THEN
                            t6 := 0;    s6 := 0;
                            IF s_blend &lt;= blend_s1 THEN    // 直接减速都超距离
                                s_blend := blend_s1;
                                blend_s2 := 0;
                                t7 := 0;
                                t8 := 0;
                                t9 := 0;
                                WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num,AddrOffset := 132,Data := blend_equal_vel);
                            ELSE    // 二分查找最大能到的混成速度
                                IF Axis_blending_v &gt; blend_equal_vel - 0.004(* MINV *) THEN  // 混承速度 &gt; 当前速度
                                    FB_S_Type_Motion_Par_Calc_Fix_S1(vs := blend_equal_vel,vm := Axis_blending_v,vt := blend_equal_vel,acc := acc,dec := acc,jerk := jerk,    // 对称，dec = acc
                                        s_all := (s_blend - blend_s1) * 2,vmax =&gt; Axis_blending_v,t1 =&gt; t7,t2 =&gt; t8,t3 =&gt; t9,t4 =&gt; ,t5 =&gt; ,t6 =&gt; ,t7 =&gt; ,
                                        s1 =&gt; s7,s2 =&gt; s8,s3 =&gt; s9,s4 =&gt; ,s5 =&gt; ,s6 =&gt; ,s7 =&gt; ,t_all =&gt; );
                                ELSE                                                       // 混承速度 &lt;= 当前速度
                                    FB_S_Type_Motion_Par_Calc_Fix_S1(vs := blend_equal_vel,vm := Axis_blending_v,vt := Axis_blending_v,acc := 0,dec := dec,jerk := jerk,
                                        s_all := (s_blend - blend_s1),vmax =&gt; Axis_blending_v,t1 =&gt; t7,t2 =&gt; t8,t3 =&gt; t9,t4 =&gt; ,t5 =&gt; ,t6 =&gt; ,t7 =&gt; ,
                                        s1 =&gt; s7,s2 =&gt; s8,s3 =&gt; s9,s4 =&gt; ,s5 =&gt; ,s6 =&gt; ,s7 =&gt; ,t_all =&gt; );
                                END_IF;

                                WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 132, Data := Axis_blending_v);
                            END_IF;
                        END_IF;

                        t_all := blend_start_t + t5 + t6 + t7 + t8 + t9;
                    ELSE                 // 梯形混成
                        IF t_now &lt; t1 + t2 THEN
                            s_blend := s_all - s1 - Motion_Delta_S(vs := vmax,acc := 0,dec := 0,jerk := 0,d_t := UDINT_TO_REAL(t_now) - t1);
                            blend_equal_vel := vmax;
                        ELSE
                            s_blend := s_all - s1 - s2 - Motion_Delta_S(vs := vmax,acc := 0,dec := dec,jerk := 0,d_t := UDINT_TO_REAL(t_now) - t1 - t2);
                            blend_equal_vel := vmax - dec * (UDINT_TO_REAL(t_now) - t1 - t2);
                        END_IF;

                        blend_start_t := t_now;
                        blend_start_pos := start_postion + s_all - s_blend;

                        Axis_blending_v := MC_RdAxisPar_FL(AxisNum := Axis_num,AddrOffset := 132);

                        IF Axis_blending_v &gt; blend_equal_vel - 0.004(* MINV *) THEN  // 混承速度 &gt; 当前速度
                            run_mode := 4(* RM_BLENDING_EQU_ACC *);
                            t5 := (Axis_blending_v - blend_equal_vel) / acc;
                            s5 := Motion_Delta_S(vs := blend_equal_vel,acc := acc,dec := 0,jerk := 0,d_t := t5);
                        ELSE                                                       // 混承速度 &lt;= 当前速度
                            run_mode := 3(* RM_BLENDING_EQU_DEC *);
                            t5 := (blend_equal_vel - Axis_blending_v) / dec;
                            s5 := Motion_Delta_S(vs := blend_equal_vel,acc := 0,dec := dec,jerk := 0,d_t := t5);
                        END_IF;

                        s4 := s_blend - s5;
                        t4 := s4 / blend_equal_vel;
                        IF t4 &lt; 0 THEN
                            t4 := 0; s4 := 0;
                            IF Axis_blending_v &gt; blend_equal_vel - 0.004(* MINV *) THEN  // 混承速度 &gt; 当前速度
                                t5 := (ESQR(blend_equal_vel * blend_equal_vel + 2 * acc * s_blend) - blend_equal_vel) / acc;
                            ELSE
                                t5 := (blend_equal_vel - ESQR(blend_equal_vel * blend_equal_vel + 2 * dec * s_blend)) / dec;
                            END_IF;
                            s5 := s_blend;
                        END_IF;
                        t_all := blend_start_t + t4 + t5;
                    END_IF;
                END_IF;
            END_IF;
        END_IF;

        WRAxisPar_Res := MC_WrAxisPar_FL(AxisNum := Axis_num, AddrOffset := 92, Data := Axis_taget_postion);
        t_now := t_now + 1;
    END_IF;

END_IF;

execute_before := Execute;

END_PROGRAM