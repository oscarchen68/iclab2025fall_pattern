`define CYCLE_TIME  20.0

module PATTERN(
    // output signals
    clk,
    rst_n,
    in_valid_data,
    in_valid_param,
    
    data,
    index,
    mode,
    QP,
    
    // input signals
    out_valid,
    out_value
);

// ========================================
// I/O declaration
// ========================================
// Output
output reg          clk;
output reg          rst_n;
output reg          in_valid_data;
output reg          in_valid_param;

output reg    [7:0] data;
output reg    [3:0] index;
output reg          mode;
output reg    [4:0] QP;

// Input
input               out_valid;
input signed [31:0] out_value;

//===========================================
// File paths
//===========================================
parameter INPUT_FILE  = "../00_TESTBED/output/input.txt";
parameter GOLDEN_FILE = "../00_TESTBED/output/golden.txt";

reg released_reset;
initial released_reset = 1'b0;

localparam integer MAX_CYCLES_WAIT_OUTVALID = 200;
localparam integer MAX_CYCLES_PER_PATTERN    = 2000;
localparam integer IDLE_STABLE_NEGEDGES      = 5;

// NEW: per-index timeout and expected outputs
localparam integer EXPECTED_OUT_PER_INDEX = 1024;
localparam integer MAX_CYCLES_PER_INDEX = 20000; // adjust if needed

// =======================================================
// input.txt format expected by this PATTERN:
//  NUM_PATTERNS <N>
//  PATTERN 1
//    FRAME 0
//    <1024 hex pixels>
//    ...
//    FRAME 15
//    <1024 hex pixels>
//    INDEX 0 MODE <m> QP <q>
//    ...
//    INDEX 15 MODE <m> QP <q>
//  ENDPATTERN
// ========================================================

// ========================================
// clock
// ========================================
real CYCLE = `CYCLE_TIME;
always #(CYCLE/2.0) clk = ~clk; // clock

// ------------------------------------------------------
// Cycle counter
// ------------------------------------------------------
integer cycle_cnt;
initial cycle_cnt = 0;
always @(posedge clk) cycle_cnt = cycle_cnt + 1;

// Record start and end cycles
integer start_cycle, end_cycle;

// reset → outputs must be zero
task reset_and_check;
begin
    rst_n           = 1'b1;
    in_valid_data   = 1'b0;
    in_valid_param  = 1'b0;
    data            = 8'bx;
    index           = 4'bx;
    mode            = 1'bx;
    QP              = 5'bx;

    force clk = 1'b0;
    #(10);
    rst_n = 1'b0; // async low
    #(2*CYCLE);
    if ((out_valid !== 1'b0) || (out_value !== 32'b0) ) begin
        show_fail;
        $display("Fail! All outputs must be 0 when reset.");
        $finish;
    end
    rst_n = 1'b1;
    #(3); release clk;
    released_reset = 1'b1;
    #(20);
end
endtask

// when out_valid==0, out_value must be zero
always @(negedge clk) begin
    if (released_reset && (out_valid === 1'b0)) begin
        if (out_value !== 32'b0) begin
            show_fail;
            $display("Fail! out_value must be 0 when out_valid is 0.");
            repeat(2) @(negedge clk);
            $finish;
        end
    end
end

// out_valid should NOT overlap with in_valid_data or in_valid_param
always @(negedge clk) begin
    if (released_reset) begin
        if ((out_valid === 1'b1) && (in_valid_data === 1'b1 || in_valid_param === 1'b1)) begin
            show_fail;
            $display("Fail! out_valid must not overlap with in_valid_data/in_valid_param.");
            repeat(2) @(negedge clk);
            $finish;
        end
    end
end

// Wait random 1..3 negedges between patterns
integer gap_negedges;
task wait_gap_2to4_negedges;
begin
    @(negedge clk);
    // gap_negedges = ($random % 3);
    // if (gap_negedges < 0) gap_negedges = -gap_negedges;
    // gap_negedges = gap_negedges + 1; // 1..3
    // repeat (gap_negedges) @(negedge clk);
end
endtask

// ===============================================================
// File & control variables
// ===============================================================
integer fin, fgold;
integer num_patterns;
integer p_id;
integer idx_i, mode_i, qp_i;
integer ret;
integer tmp_int;
integer pixel_cnt;
integer golden_expected;
integer golden_counter;
integer case_id;
integer frame_id;

reg [7:0] frame_mem [0:15][0:1023]; // 16 frames * 1024 pixels

// NEW: per-index waiting control
reg index_waiting;
integer idx_out_count; // how many out_valid pulses seen for current index

// golden checking: compare DUT out_value with golden.txt
always @(posedge clk) begin
    if (released_reset) begin
        if (out_valid === 1'b1) begin
            // read next golden expected value (global order)
            ret = $fscanf(fgold, "%d\n", golden_expected);
            if (ret != 1) begin
                show_fail;
                $display("ERROR: golden file ended prematurely at golden_counter=%0d", golden_counter);
                $finish;
            end
            golden_counter = golden_counter + 1;

            // check DUT output
            if (out_value !== $signed(golden_expected)) begin
                show_fail;
                $display("Mismatch at golden index %0d", golden_counter-1);
                $display(" DUT out_value = %0d (0x%08h)", out_value, out_value);
                $display(" Expected      = %0d (0x%08h)", golden_expected, golden_expected);
                repeat (5) @(negedge clk);
                $finish;
            end

            // if we're waiting for the current INDEX's outputs, count them
            if (index_waiting) begin
                idx_out_count = idx_out_count + 1;
            end
        end
    end
end

// Wait until out_valid==0 and it remains 0 for 'stable_negedges' consecutive negedges.
// Fail if we waited more than max_cycles.
task wait_for_idle_stable_with_timeout;
    input integer max_cycles;
    input integer stable_negedges;
    input [511:0] msg;
    integer waited;
    integer stable_cnt;
begin
    waited = 0;
    stable_cnt = 0;
    while (1) begin
        @(negedge clk);
        waited = waited + 1;
        if (out_valid === 1'b0) stable_cnt = stable_cnt + 1;
        else stable_cnt = 0;
        if (stable_cnt >= stable_negedges) disable wait_for_idle_stable_with_timeout;
        if (waited > max_cycles) begin
            show_fail;
            $display("Timeout waiting for DUT idle. (%s)", msg);
            $finish;
        end
    end
end
endtask

// NEW: wait for this index to produce 'expected' out_valid pulses (not necessarily continuous)
task wait_for_index_outputs;
    input integer expected;
    input integer max_cycles;
    integer waited;
begin
    waited = 0;
    while (idx_out_count < expected) begin
        @(negedge clk);
        waited = waited + 1;
        if (waited > max_cycles) begin
            show_fail;
            $display("Timeout waiting for index outputs: got %0d / %0d within %0d cycles", idx_out_count, expected, max_cycles);
            $finish;
        end
    end
    // done for this index
    index_waiting = 1'b0;
end
endtask

// ===============================================================
// Main
// ===============================================================
initial begin
    golden_counter = 0;
    released_reset = 1'b0;
    case_id        = 0;
    index_waiting  = 1'b0;
    idx_out_count  = 0;

    // Open files
    fin   = $fopen(INPUT_FILE,  "r");
    fgold = $fopen(GOLDEN_FILE, "r");
    if (fin==0 || fgold==0) begin
        $display("Cannot open input/golden files. Check paths.");
        $finish;
    end

    // Reset
    reset_and_check();
    @(negedge clk);

    // Read number of patterns
    ret = $fscanf(fin, "NUM_PATTERNS %d\n", num_patterns);
    if (ret != 1) begin
        $display("Input file format error: cannot read NUM_PATTERNS");
        $finish;
    end
    $display("PATTERN: NUM_PATTERNS = %0d", num_patterns);

    // For each pattern
    for (p_id = 1; p_id <= num_patterns; p_id = p_id + 1) begin
        // read PATTERN header
        ret = $fscanf(fin, "PATTERN %d\n", tmp_int);
        if (ret != 1) begin
            $display("Input file format error: expected PATTERN %0d header", p_id);
            $finish;
        end
        $display("PATTERN: Starting pattern %0d", p_id);
        start_cycle = cycle_cnt;

        // --------------------------
        // (1) Read 16 frames (32x32 each) into frame_mem
        // --------------------------
        for (frame_id = 0; frame_id < 16; frame_id = frame_id + 1) begin
            ret = $fscanf(fin, "FRAME %d\n", tmp_int);
            if (ret != 1) begin
                $display("Missing FRAME header at frame %0d", frame_id);
                $finish;
            end
            for (pixel_cnt = 0; pixel_cnt < 1024; pixel_cnt = pixel_cnt + 1) begin
                ret = $fscanf(fin, "%h\n", tmp_int);
                if (ret != 1) begin
                    $display("Pixel read error frame %0d pixel %0d", frame_id, pixel_cnt);
                    $finish;
                end
                frame_mem[frame_id][pixel_cnt] = tmp_int[7:0];
            end
        end

        // --------------------------
        // (2) SEND ALL FRAMES to DUT (in_valid_data pulses)
        //     We stream frame 0..15 sequentially, 1024 cycles each.
        // --------------------------
        // Make sure DUT not currently asserting out_valid
        while (out_valid === 1'b1) @(negedge clk);

        for (frame_id = 0; frame_id < 16; frame_id = frame_id + 1) begin
            for (pixel_cnt = 0; pixel_cnt < 1024; pixel_cnt = pixel_cnt + 1) begin
                @(negedge clk);
                data <= frame_mem[frame_id][pixel_cnt];
                in_valid_data <= 1'b1;
            end
        end
        // finish data burst
        @(negedge clk);
        in_valid_data <= 1'b0;
        data <= 'dx;

        // Wait a small gap to stabilize before sending params
        wait_gap_2to4_negedges;

        // --------------------------
        // (3) Read 16 parameter lines and for each:
        //     - send 1-cycle param pulse
        //     - wait for DUT to produce EXPECTED_OUT_PER_INDEX outputs (may be non-contiguous)
        // --------------------------
        for (idx_i = 0; idx_i < 16; idx_i = idx_i + 1) begin
            integer m0, m1, m2, m3;
            ret = $fscanf(fin, "INDEX %d MODES %d %d %d %d QP %d\n", tmp_int, m0, m1, m2, m3, qp_i);
            if (ret != 6) begin
                $display("Missing INDEX line for index %0d", idx_i);
                $finish;
            end

            // Wait until DUT not asserting out_valid (avoid overlapping param with output)
            while (out_valid === 1'b1) @(negedge clk);

            // Prepare to wait for outputs of this index
            idx_out_count = 0;
            index_waiting = 1'b1;

            // Send parameter pulse (place values at negedge so DUT samples at posedge)
            @(negedge clk);
            index <= tmp_int[3:0];
            mode  <= m0[0];
            QP    <= qp_i[4:0];
            in_valid_param <= 1'b1;
            
            @(negedge clk);
            index <= 'bx;
            mode  <= m1[0];
            QP    <= 'bx;

            @(negedge clk);
            mode  <= m2[0];

            @(negedge clk);
            mode  <= m3[0];

            @(negedge clk);
            mode <= 'bx;
            in_valid_param <= 1'b0;

            // Now wait for the DUT to emit EXPECTED_OUT_PER_INDEX outputs for this index
            wait_for_index_outputs(EXPECTED_OUT_PER_INDEX, MAX_CYCLES_PER_INDEX);

            // After this returns, we have seen 1024 outputs for this index and they were checked already by golden check
            // safe to proceed to next index
        end

        // optionally consume ENDPATTERN line
        ret = $fscanf(fin, "ENDPATTERN\n");

        // Wait DUT becomes idle (out_valid==0 stable)
        //wait_for_idle_stable_with_timeout(MAX_CYCLES_PER_PATTERN, IDLE_STABLE_NEGEDGES, "waiting for DUT idle after pattern");

        // gap
        //wait_gap_2to4_negedges();

        end_cycle = cycle_cnt;
        $display("PATTERN %0d finished (cycles used: %0d)", p_id, end_cycle - start_cycle);
    end // pattern loop

    // After all patterns processed: wait a few cycles then finish
    repeat (10) @(negedge clk);

    // check golden file fully consumed (optional)
    ret = $fscanf(fgold, "%d\n", tmp_int);
    if (ret != -1) $display("Warning: golden file has extra lines after expected outputs.");

    show_success();
    $display("All patterns passed. Total golden outputs checked: %0d", golden_counter);
    $finish;
end

endmodule


task show_success;
begin
$display("⠀⠀⠀⠀⠀⠀⣀⣀⠀⠀⠀⠀⠀⠀⣀⣤⠶⠞⠛⠉⠉⠉⠙⠛⠲⠶⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⢀⣾⠿⠟⠙⠿⠛⣷⣀⣴⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠳⣦⡀⢀⣤⣤⡶⣦⣤⡀⠀⠀⠀");
$display("⠀⠀⠀⢠⣿⠂⠀⠀⠀⢀⣾⠏⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⠉⠉⠀⠀⢸⡇⠀⠀⠀");
$display("⠀⠀⠀⠈⠛⣶⠀⠀⢠⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣧⡀⠀⢠⣴⠟⠀⠀⠀");
$display("⠀⠀⢀⡀⣀⣼⣛⢲⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⢶⣶⣧⣀⣀⠀⠀");
$display("⢠⣴⣿⠉⠛⠀⠉⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣟⠁⠘⠋⠹⢷⡄");
$display("⠸⣷⡄⠀⠀⠀⢰⡟⠀⠀⠀⠀⠀⣰⣶⣄⠀⠀⣀⠀⠀⡀⠀⢀⡀⠀⢰⣿⣷⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⣾⠃");
$display("⠀⠛⠿⣷⣀⣀⣼⣇⠀⠀⠀⠄⣀⠻⡿⠋⠀⠀⠻⣦⡾⠻⠶⠾⠃⠀⠈⠛⠛⡀⠀⠉⠳⠄⣿⣦⣄⣽⠟⠋⠀");
$display("⠀⠀⠀⣈⣽⡟⠳⣿⡀⣇⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢷⣀⠀⣠⢀⡿⠺⢿⣅⣀⠀⠀");
$display("⠀⠀⣸⣯⠉⠁⠀⠸⣧⠘⠶⠴⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⢁⣾⠃⠀⠈⢩⣿⡀⠀");
$display("⠀⠀⠿⣶⡀⣤⠀⢀⡿⣷⣤⣤⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣴⢿⣇⢀⣰⡆⣼⡾⠏⠀");
$display("⠀⠀⠀⠘⠛⠛⠷⠛⢹⡏⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢻⡟⣿⡷⠟⠛⠛⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⣈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⠿⠋⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⢀⣴⠶⢶⠟⢻⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⢠⡾⡟⠀⠀⠀⠸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠃⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠘⠷⣦⣀⣰⡀⢠⡿⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠷⣦⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠈⠉⠉⠛⠛⢷⣤⣀⣠⣦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣴⣟⣁⣴⠟⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠉⠉⠛⠛⠓⠶⠶⠶⠶⠖⠛⠛⠉⠁⠀⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀");
end
endtask

task show_fail;
begin
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⠴⠶⠶⠶⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⣿⣽⡖⠂⠀⠀⠀⠀⠉⠙⠳⢦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣽⣾⠏⠀⠀⠀⠀⠀⠀⠻⢿⣯⣗⣤⣉⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⡿⠟⠓⠾⣿⡀⠀⠀⠀⠄⠀⠘⢿⣟⢿⣿⣦⡌⡻⣦⡀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⣿⡏⣴⣦⣤⣦⡊⣿⣦⣤⣀⣀⣀⠀⢈⣿⣾⣷⣽⣿⠶⠿⠇⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣷⣿⣷⣼⠅⠁⣯⣡⣿⠶⠶⠾⣿⣿⣿⣿⡛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⢠⡾⠿⣯⣿⣿⣿⣿⣿⣿⣿⡿⣯⡄⡄⢀⠀⢉⡝⠿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⣠⡟⠘⣾⣯⣽⡿⠿⠟⠛⠛⠛⠛⠻⢶⣅⡀⠀⣛⢻⣿⡺⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⣀⣙⣉⣿⠟⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠳⠶⠾⠿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⢿⣟⣯⣭⣭⣽⠇⠀⠀⠀⢀⣤⡀⠀⠀⠀⠀⠀⠀⣠⣤⣀⠀⠀⠀⢻⡏⣭⣉⣉⣿⡷⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⣠⠾⣫⣽⠆⡏⠀⠀⠀⠀⠰⣿⡇⠀⠀⠀⠀⠀⠐⢾⣿⠋⠀⠀⠀⠈⣇⠰⣮⡙⢶⣄⠀⠀⠀⠀⠀⠀⠀");
$display("⠘⠛⣿⣟⣩⡾⢧⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⢀⡿⢦⣈⢻⡟⠟⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠻⠛⠉⠀⠘⢷⡀⠀⠀⠀⠀⠀⠈⠙⢿⠟⠙⠁⠀⠀⠀⠀⠀⣠⡾⠀⠀⠙⠻⡟⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠈⠙⢶⣤⣀⠀⠀⡀⠀⠀⠀⡀⢀⡀⢀⣔⣶⠾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠉⠙⠛⠃⠀⠀⠀⠀⠘⠋⠉⠁⠙⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠆⠀⠀⠸⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⣰⠇⠀⠀⢀⣄⠀⠀⠀⠀⢀⡾⠁⠀⠀⠀⠀⢹⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⣿⢀⣴⣴⣟⠀⠀⠀⠀⠀⣼⠀⠀⣀⣤⠖⠀⠀⢿⡀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣄⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠛⠁⢸⣆⠀⠀⠀⠀⠈⠙⠛⠉⠀⠀⠀⠀⠀⠻⣶⣶⠚⠋⠙⠻⠾⠯⠁⢹⡇⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣦⡀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠈⠿⠛⠛⠒⠒⢻⠀⣄⣸⡇⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⢻⡀⠀⠀⢸⠆⠀⠀⠀⠀⣠⡟⢀⣼⠟⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⠦⣤⡀⢸⡈⢷⡀⢀⡿⠀⠀⣀⣠⣾⣯⡶⠛⠁⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⢠⣰⡿⢶⣽⣿⣶⣾⣿⣿⡾⠟⠋⠀⠀⠀⠀⠀⠀");
$display("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠋⠀⠀⠀⠉⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
end
endtask



