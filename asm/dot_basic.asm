;
; -------------------------------------------------------------------------------------
; DOT PRODUCT USING SCALAR REGISTERS
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
; - loop over input values with scalar float multiplication
; -------------------------------------------------------------------------------------
;

.CODE

dot_basic PROC
    ;
    ; function entry state:
    ; rcx       - pointer to input vectors
    ; rdx       - input size (2 x vector length)
    ;
    ; used regs:
    ; r10       - loop counter
    ; r11       - vector len
    ; r9        - b vector start pointer
    ; xmm0      - result accumulator
    ; xmm1      - a float value
    ; xmm2      - b float value
    ;

    PXOR xmm0, xmm0                                 ; clear accumulator
    XOR r10, r10                                    ; clear loop counter

    MOV r11, rdx                                    ; r11 will hold len of single vector
    SHR r11, 01                                     ; div rdx by 2

    LEA r9, [rcx + r11 * 4]                         ; r9 will hold b vector start -> a + (vectors len / 2) * 4

    dot_loop_start:
        CMP r10, r11                                ; compare counter with vector length
        JAE dot_loop_done                           ; finish if equal

        MOVSS xmm1, DWORD PTR [rcx + r10 * 4]       ; copy a vector element
        MOVSS xmm2, DWORD PTR [r9  + r10 * 4]       ; copy b vector element

        MULSS xmm1, xmm2                            ; multiply a x b element
        ADDSS xmm0, xmm1                            ; add sum to accumulator

        INC r10
        JMP dot_loop_start
    dot_loop_done:

    VZEROUPPER                                      ; cleanup
    RET

dot_basic ENDP

END
