PROGRAM MAIN
VAR
    axis : ARRAY[1..2] OF INT;
    pt   : STRUCT
        X : INT;
        Y : INT;
    END_STRUCT;
END_VAR

axis[1] := 10;
pt.X := axis[1];
END_PROGRAM
