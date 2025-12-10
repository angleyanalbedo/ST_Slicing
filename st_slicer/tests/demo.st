PROGRAM MAIN
VAR
    x : INT;
    y : INT;
END_VAR

x := 1;
IF x = 1 THEN
    x := 2;
END_IF;

FOR y:=1 TO 5 DO
    x := x + y;
END_FOR;

END_PROGRAM
