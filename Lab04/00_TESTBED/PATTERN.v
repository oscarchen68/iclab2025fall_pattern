`timescale 1ns/1ps
`define CYCLE_TIME      45.0

module PATTERN(
    // Output Port
    clk,
    rst_n,
    in_valid,
    Image,
    Kernel_ch1,
    Kernel_ch2,
	Weight_Bias,
    task_number,
    mode,
    capacity_cost,
    // Input Port
    out_valid,
    out
    );

//---------------------------------------------------------------------
//   PORT DECLARATION          
//---------------------------------------------------------------------
output reg        clk, rst_n, in_valid;
output reg [31:0]  Image;
output reg [31:0]  Kernel_ch1;
output reg [31:0]  Kernel_ch2;
output reg [31:0]  Weight_Bias;
output reg        task_number;
output reg [1:0]   mode;
output reg [3:0]   capacity_cost;

input           out_valid;
input   [31:0]  out;

//---------------------------------------------------------------------
//   PARAMETER & INTEGER DECLARATION
//---------------------------------------------------------------------
real CYCLE = `CYCLE_TIME;
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch_type = 0;
parameter inst_arch = 0;



parameter INPUT_FILE = "../00_TESTBED/input.txt";
parameter GOLDEN_FILE = "../00_TESTBED/golden.txt";

integer fin, fgold;
integer ret;

// book-keeping
integer case_id;

reg released_reset;
initial released_reset = 1'b0;

// Random gap between cases: 2~4 negedges
integer gap_negedges;

// For reading tokens
integer i;
integer tk_task, tk_mode;
reg [31:0] tmp32, tmp32_b;
integer tmp4;

// For golden compare buffers
reg [31:0] golden32 [0:2]; // Task0: 3 outputs; Task1: use golden32[0]

// ===============================================================
// Clock
// ===============================================================
initial clk = 1'b0;
always  #(CYCLE/2.0) clk = ~clk;

// ------------------------------------------------------
// Cycle counter
// ------------------------------------------------------
integer cycle_cnt;
initial cycle_cnt = 0;
always @(posedge clk) cycle_cnt = cycle_cnt + 1;

// Record start and end cycles for each case
integer start_cycle, end_cycle;


// SPEC-4: reset → outputs must be zero
task reset_and_check;
begin
    rst_n           = 1'b1;
    in_valid        = 1'b0;
    task_number     = 'bx;
    mode            = 'bx;
    Image           = 'bx;
    Kernel_ch1      = 'bx;
    Kernel_ch2      = 'bx;
    Weight_Bias     = 'bx;
    capacity_cost   = 'bx;


    force clk = 1'b0;

    #(10);
    rst_n = 1'b0; // async low
    #(2*CYCLE);
    if ((out_valid !== 1'b0) || (out !== 32'b0) ) begin
        $display("---------------------------------------------------------------------------------------------");
        $display("             Fail! All outputs set to 0 when reset.");
        $display("---------------------------------------------------------------------------------------------");
        $display("                    SPEC-4 FAIL                   ");
        $finish;
    end
    rst_n = 1'b1;
    #(3);  release clk;
    released_reset = 1'b1;
    #(20);
end
endtask

// SPEC-5: when out_valid==0, outputs must be zero
always @(negedge clk) begin
    if (released_reset && (out_valid===1'b0)) begin
        if (out !== 10'd0) begin
            $display("---------------------------------------------------------------------------------------------");
            $display("             Fail! Output value set to 0 when out_valid is 0.");
            $display("---------------------------------------------------------------------------------------------");
            $display("                    SPEC-5 FAIL                   ");
            repeat(2) @(negedge clk);
            $finish;
        end
    end
end

// out_valid should NOT overlap with in_valid
always @(negedge clk) begin
    if (released_reset && (in_valid===1'b1) && (out_valid===1'b1)) begin
        $display("---------------------------------------------------------------------------------------------");
        $display("             Fail! Out_valid should not overlap with in_valid.");
        $display("---------------------------------------------------------------------------------------------");
        $display("                    FAIL                   ");
        repeat(2) @(negedge clk);
        $finish;
    end
end

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------
// Convert IEEE-754 single precision (bits) -> real number
// (implemented purely in Verilog since 'shortreal' is unavailable)
function real fp32_to_real;
    input [31:0] x;
    integer s; integer e; integer k;
    real m, val, p2;
begin
    s = x[31];
    e = x[30:23];
    m = 0.0;
    // fraction
    for (k=0; k<23; k=k+1) begin
        if (x[k]) m = m + (1.0 / (2.0 ** (23-k)));
    end
    if (e==0) begin
        // subnormal: exponent = 1-127, mantissa = 0.f
        val = (m) * (2.0 ** (-126));
    end else if (e==255) begin
        // Inf/NaN: treat as huge number to force mismatch if occurs
        val = 1.0e300;
    end else begin
        // normal: 1.f
        val = (1.0 + m) * (2.0 ** (e-127));
    end
    if (s) val = -val;
    fp32_to_real = val;
end
endfunction

function real fabs; input real r; begin fabs = (r>=0.0)? r : -r; end endfunction

// Wait random 2~4 negedges between cases (SPEC-8)
task wait_gap_2to4_negedges;
begin
    gap_negedges = ($random % 3);
    if (gap_negedges < 0) gap_negedges = -gap_negedges;
    gap_negedges = gap_negedges + 1; // 1..3
    repeat (gap_negedges) @(negedge clk);
    //repeat (1) @(negedge clk);
end
endtask



// ===============================================================
// Core: wait for out_valid, measure latency, collect outputs, and compare
//   - SPEC-6: Latency ≤ 150 cycles (counted after in_valid deasserts)
//   - Task0: Expect 3 consecutive floating-point outputs (tolerance < 1e-6)
//   - Task1: Expect 1 output (32-bit mask, bitwise match)
// ===============================================================
task wait_and_collect_outputs_and_check;
    integer wait_cycles, out_cnt;
    real    out_r, golden_r, err;
begin
    wait_cycles   = 0;
    out_cnt       = 0;
    start_cycle = cycle_cnt;

    // Wait until the first out_valid=1 (start latency measurement)
    while (out_valid !== 1) begin
        wait_cycles = wait_cycles + 1;
        if (wait_cycles > 150) begin
            $display("SPEC-6 FAIL: execution latency > 150 cycles (case=%0d)", case_id);
            $finish;
        end
        @(negedge clk);
    end

    if (tk_task==0) begin
        // ----- Task0: expect 3 consecutive outputs -----
        repeat (3) begin
            out_r    = fp32_to_real(out);
            golden_r = fp32_to_real(golden32[out_cnt]);
            err = fabs(out_r - golden_r);
            if (err > 1.0e-6) begin
                $display("FAIL (Task0) case=%0d idx=%0d | out=%h got=%.9f exp=%.9f | err=%.3e",
                         case_id, out_cnt, out, out_r, golden_r, err);
                $finish;
            end
            out_cnt = out_cnt + 1;
            @(negedge clk);
            if (out_cnt<3 && out_valid!==1) begin
                $display("FAIL (Task0) out_valid must be 3 consecutive cycles (case=%0d)", case_id);
                $finish;
            end
        end
        // Must drop back to 0 on the next cycle
        if (out_valid!==0) begin
            $display("FAIL (Task0) out_valid should drop after 3 cycles (case=%0d)", case_id);
            $finish;
        end
    end else begin
        // ----- Task1: expect 1 output -----
        if (out!==golden32[0]) begin
            $display("FAIL (Task1) case=%0d | out=%h expected=%h", case_id, out, golden32[0]);
            $finish;
        end
        @(negedge clk);
        if (out_valid!==0) begin
            $display("FAIL (Task1) out_valid must be exactly 1 cycle (case=%0d)", case_id);
            $finish;
        end
    end
    end_cycle = cycle_cnt;
    $display("[Task%0d] case %0d finished at cycle %0d, latency=%0d cycles",
             tk_task, case_id, end_cycle, end_cycle-start_cycle);

end
endtask


// ----------------------
// buffers for input staging (放在 module scope)
// ----------------------
reg [31:0] buf_img   [0:71];
reg [31:0] buf_ker1  [0:17];
reg [31:0] buf_ker2  [0:17];
reg [31:0] buf_bias  [0:56];
reg [3:0]  buf_cap   [0:4];

// ===============================================================
// Send a Task0 transaction:
//   1. Load Image/Kernel/Bias into buffers
//   2. Use a single loop to drive inputs (each stream set to 'bx after use)
// ===============================================================
task drive_task0_and_check;
    integer idx;
begin
    $display("[Task0] case %0d start to send input at cycle %0d", case_id, cycle_cnt);
    // Step 1: Read all data from file (72 image, 18 (ker1 ker2), 57 bias)
    for (idx=0; idx<72; idx=idx+1) begin
        ret = $fscanf(fin, "%h\n", buf_img[idx]);
        if (ret!=1) begin $display("INPUT EOF/format error (Task0 Image read)"); $finish; end
    end
    for (idx=0; idx<18; idx=idx+1) begin
        ret = $fscanf(fin, "%h %h\n", buf_ker1[idx], buf_ker2[idx]);
        if (ret!=2) begin $display("INPUT EOF/format error (Task0 Kernel read)"); $finish; end
    end
    for (idx=0; idx<57; idx=idx+1) begin
        ret = $fscanf(fin, "%h\n", buf_bias[idx]);
        if (ret!=1) begin $display("INPUT EOF/format error (Task0 Bias read)"); $finish; end
    end

    // Step 2: drive signals for 72 cycles
    // All inputs updated per cycle; unused streams set to 'bx
    for (idx=0; idx<72; idx=idx+1) begin
        // Image always valid for idx<72
        Image = buf_img[idx];

        // Kernel valid only for idx < 18
        if (idx < 18) begin
            Kernel_ch1 = buf_ker1[idx];
            Kernel_ch2 = buf_ker2[idx];
        end else begin
            Kernel_ch1 = 'bx;
            Kernel_ch2 = 'bx;
        end

        // Weight_Bias valid only for idx < 57
        if (idx < 57) begin
            Weight_Bias = buf_bias[idx];
        end else begin
            Weight_Bias = 'bx;
        end

        // capacity_cost not used in Task0 -> keep 'bx every cycle
        capacity_cost = 'bx;

        // control signals: task_number, mode, in_valid
        task_number = (idx < 1) ? 1'b0 : 'bx;       // header semantics: only first cycle drives task_number/mode valid
        mode        = (idx < 1) ? tk_mode[1:0] : 'bx;
        in_valid    = 1'b1;

        @(negedge clk);
    end

    // End of transmission: set inputs to unknown and deassert in_valid
    in_valid      = 1'b0;
    task_number   = 'bx;
    mode          = 'bx;
    Image         = 'bx;
    Kernel_ch1    = 'bx;
    Kernel_ch2    = 'bx;
    Weight_Bias   = 'bx;
    capacity_cost = 'bx;

    // Read 3-line golden results
    for (idx=0; idx<3; idx=idx+1) begin
        ret = $fscanf(fgold, "%h\n", golden32[idx]);
        if (ret!=1) begin $display("GOLDEN EOF/format error (Task0 Golden)"); $finish; end
    end

    // Wait for output and perform latency/accuracy checks
    wait_and_collect_outputs_and_check();
end
endtask


// ===============================================================
// Send a Task1 transaction:
//   1. Load Image/Kernel/Capacity into buffers
//   2. Use a single loop to drive inputs (each stream set to 'bx after use)
// ===============================================================
task drive_task1_and_check;
    integer idx;
begin
    $display("[Task1] case %0d start to send input at cycle %0d", case_id, cycle_cnt);
    // Read: 36 image, 18 (ker1 ker2), 5 capacity_cost (decimal)
    for (idx=0; idx<36; idx=idx+1) begin
        ret = $fscanf(fin, "%h\n", buf_img[idx]);
        if (ret!=1) begin $display("INPUT EOF/format error (Task1 Image read)"); $finish; end
    end
    for (idx=0; idx<18; idx=idx+1) begin
        ret = $fscanf(fin, "%h %h\n", buf_ker1[idx], buf_ker2[idx]);
        if (ret!=2) begin $display("INPUT EOF/format error (Task1 Kernel read)"); $finish; end
    end
    for (idx=0; idx<5; idx=idx+1) begin
        ret = $fscanf(fin, "%d\n", tmp4);
        if (ret!=1) begin $display("INPUT EOF/format error (Task1 capacity read)"); $finish; end
        buf_cap[idx] = tmp4[3:0];
    end

    // single loop length = Image length (36)
    for (idx=0; idx<36; idx=idx+1) begin
        // Image valid for idx < 36
        Image = buf_img[idx];

        // Kernel valid for idx < 18
        if (idx < 18) begin
            Kernel_ch1 = buf_ker1[idx];
            Kernel_ch2 = buf_ker2[idx];
        end else begin
            Kernel_ch1 = 'bx;
            Kernel_ch2 = 'bx;
        end

        // Weight_Bias not used in Task1 -> set 'bx
        Weight_Bias = 'bx;

        // capacity_cost valid for idx < 5
        if (idx < 5) begin
            capacity_cost = buf_cap[idx];
        end else begin
            capacity_cost = 'bx;
        end

        // control signals
        task_number = (idx < 1) ? 1'b1 : 'bx;
        mode        = (idx < 1) ? tk_mode[1:0] : 'bx;
        in_valid    = 1'b1;

        @(negedge clk);
    end

    // End of transmission: set inputs to unknown and deassert in_valid
    in_valid      = 1'b0;
    task_number   = 'bx;
    mode          = 'bx;
    Image         = 'bx;
    Kernel_ch1    = 'bx;
    Kernel_ch2    = 'bx;
    Weight_Bias   = 'bx;
    capacity_cost = 'bx;

    // Read 1-line golden result
    ret = $fscanf(fgold, "%h\n", golden32[0]);
    if (ret!=1) begin $display("GOLDEN EOF/format error (Task1 Golden)"); $finish; end

    // Wait for output and perform latency/accuracy checks
    wait_and_collect_outputs_and_check();
end
endtask


// ===============================================================
// Main
// ===============================================================
initial begin
    released_reset = 1'b0;
    case_id        = 0;

    // Open input and golden files
    fin   = $fopen(INPUT_FILE,  "r");
    fgold = $fopen(GOLDEN_FILE, "r");
    if (fin==0 || fgold==0) begin
        $display("Cannot open input/golden files. Check paths:\n  %s\n  %s", INPUT_FILE, GOLDEN_FILE);
        $finish;
    end

    // Reset
    reset_and_check();
    @(negedge clk);

    // Loop through test cases until EOF
    forever begin
        ret = $fscanf(fin, "%d %d\n", tk_task, tk_mode);
        if (ret!=2) begin
            $display("PASS: All %0d test cases passed!", case_id);
            $finish;
        end

        case_id = case_id + 1;

        // header sanity
        if (tk_task!=0 && tk_task!=1) begin
            $display("INPUT format error: task must be 0/1, got %0d", tk_task);
            $finish;
        end
        if (tk_mode<0 || tk_mode>3) begin
            $display("INPUT format error: mode must be 0..3, got %0d", tk_mode);
            $finish;
        end

        if (tk_task==0) drive_task0_and_check();
        else            drive_task1_and_check();

        // SPEC-8: wait 2~4 negative edges between consecutive inputs
        wait_gap_2to4_negedges();
    end
end

endmodule
