;
; -------------------------------------------------------------------------------------
; DOT PRODUCT USING VECTOR REGISTERS - YMM/SIMD LOOP
;
; Computes dot product of two float vectors
;
; Calling convention: Windows x64
;
; Params:
; - pointer to array containing two vectors (vectorA vectorB)
; - array length in floats (2 × vector_length, must be even)
; Return:
; - dot product result float saved in XMM0
;
; Algorithm:
; - loop over array chunks loaded into ymm registers using
;   4 ymm accumulators each with 8 float lanes
; - handle remaining tail elements (<32) using scalar operation
;
; Uses:
; ymm0–ymm3  - 4 x 256 bit accumulators
; ymm4–ymm7  - 32 float values from vector A for each iteration
; ymm8-ymm11 - 32 float values from vector B for each iteration
; -------------------------------------------------------------------------------------
;

.CODE

dot_ymm PROC
    ;
    ; function entry state:
    ; rcx        - pointer to input vectors
    ; rdx        - input size (2 x vector length)
    ;
    ; used regs:
    ; r10        - loop counter
    ; r11        - vector len
    ; r9         - b vector start pointer
    ; r12        - number of loops
    ; r13        - tail len
    ; r14        - offset
    ;
    ; (4 regs - 8 floats each - 32 floats total - 128 bytes total)
    ; ymm0–ymm3  - 4 accumulators
    ; ymm4–ymm7  - a float value
    ; ymm8-ymm11 - b float value
    ;

    ; store ymm registers
    PUSH r12
    PUSH r13
    PUSH r14
    PUSH r15                                        ; for alignment only

    sub rsp, 32 * 6                                 ; space for ymm6–ymm11 (6 registers)

    VMOVDQU YMMWORD PTR [rsp +   0], ymm6
    VMOVDQU YMMWORD PTR [rsp +  32], ymm7
    VMOVDQU YMMWORD PTR [rsp +  64], ymm8
    VMOVDQU YMMWORD PTR [rsp +  96], ymm9
    VMOVDQU YMMWORD PTR [rsp + 128], ymm10
    VMOVDQU YMMWORD PTR [rsp + 160], ymm11

    VPXOR ymm0, ymm0, ymm0                          ; clear accumulators
    VPXOR ymm1, ymm1, ymm1
    VPXOR ymm2, ymm2, ymm2
    VPXOR ymm3, ymm3, ymm3

    XOR r10, r10                                    ; clear loop counter

    MOV r11, rdx                                    ; r11 will hold len of a single vector
    SHR r11, 01                                     ; div rdx by 2

    ; compute nr of loops (one load into 4 regs takes 32 floats - so we need len // 32)
    MOV r12, r11
    SHR r12, 5h                                     ; shift right by 5 bits

    ; compute tail for scalar loop after main loop is done
    MOV r13, r11
    AND r13, 1Fh                                    ; get last 5 bits

    LEA r9, [rcx + r11 * 4]                         ; r9 will hold b vector start -> a + (vectors len / 2) * 4

    ; each loop for each vector loads 4 256 regs -> 4 x 8 floats -> 1024 bits -> stride = 128 bytes
    dot_loop_start:
        PREFETCHT0 [rcx + r8 + 1024]                ; prefetch  gives minor speed improvement
        PREFETCHT0 [r9  + r8 + 1024]

        CMP r10, r12                                ; compare counter with vector length
        JAE dot_loop_done                           ; finish if equal

        MOV r8, r10                                 ; compute offset - copy current counter
        SHL r8, 7h                                  ; multiply counter x 128

        VMOVUPS ymm4,  YMMWORD PTR [rcx + r8]
        VMOVUPS ymm5,  YMMWORD PTR [rcx + r8 + 1 * 32]
        VMOVUPS ymm6,  YMMWORD PTR [rcx + r8 + 2 * 32]
        VMOVUPS ymm7,  YMMWORD PTR [rcx + r8 + 3 * 32]

        VMOVUPS ymm8,  YMMWORD PTR [r9 + r8]
        VMOVUPS ymm9,  YMMWORD PTR [r9 + r8 + 1 * 32]
        VMOVUPS ymm10, YMMWORD PTR [r9 + r8 + 2 * 32]
        VMOVUPS ymm11, YMMWORD PTR [r9 + r8 + 3 * 32]

        VFMADD231PS ymm0, ymm4, ymm8
        VFMADD231PS ymm1, ymm5, ymm9
        VFMADD231PS ymm2, ymm6, ymm10
        VFMADD231PS ymm3, ymm7, ymm11

        INC r10
        JMP dot_loop_start
    dot_loop_done:

        ; combine accumulators
        VADDPS ymm0, ymm0, ymm1                     ; 0 + 1
        VADDPS ymm2, ymm2, ymm3                     ; 2 + 3
        VADDPS ymm0, ymm0, ymm2                     ; 0 + 1 + 2 + 3

        ; reduce to xmm0[0]
        VEXTRACTF128 xmm1, ymm0, 1                  ; extract upper bits (lane 1) of ymm0 to xmm1
        VADDPS  xmm0, xmm0, xmm1                    ; add extracted xmm1 to lane 0 of ymm0 -> xmm0 holds 4 floats now
        VHADDPS xmm0, xmm0, xmm0                    ; horizontal add -> reduced to sum of 1, 2 in xmm0[0]
        VHADDPS xmm0, xmm0, xmm0                    ; horizontal add -> reduced to sum of 1, 2, 3, 4 in xmm0[0]

        XOR r10, r10                                ; clear loop counter

        ; compute offset for tail loop
        MOV r14, r12                                ; copy nr of iterations done
        SHL r14, 7h                                 ; mul by iter size (128 bytes), now r14 holds offset

        ; move start pointers by offset
        ADD rcx, r14
        ADD r9, r14


    tail_loop_start:
        CMP r10, r13                                ; compare counter with tail length
        JAE tail_loop_done                          ; finish if equal

        MOVSS xmm1, DWORD PTR [rcx]                 ; copy a vector element
        MOVSS xmm2, DWORD PTR [r9]                  ; copy b vector element

        MULSS xmm1, xmm2                            ; multiply a x b element
        ADDSS xmm0, xmm1                            ; add sum to accumulator

        INC r10                                     ; increase counter

        ; move start pointers by stride size
        ADD rcx, 4h
        ADD r9, 4h

        JMP tail_loop_start
    tail_loop_done:

    ; restore ymm registers
    VMOVDQU ymm6,  YMMWORD PTR [rsp +   0]
    VMOVDQU ymm7,  YMMWORD PTR [rsp +  32]
    VMOVDQU ymm8,  YMMWORD PTR [rsp +  64]
    VMOVDQU ymm9,  YMMWORD PTR [rsp +  96]
    VMOVDQU ymm10, YMMWORD PTR [rsp + 128]
    VMOVDQU ymm11, YMMWORD PTR [rsp + 160]
    ADD rsp, 32 * 6

    ; cleanup
    POP r15
    POP r14
    POP r13
    POP r12

    VZEROUPPER

    RET

dot_ymm ENDP

END
