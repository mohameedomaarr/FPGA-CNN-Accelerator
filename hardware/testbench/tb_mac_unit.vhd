
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity tb_mac_unit is
end entity;


architecture simulation of tb_mac_unit is

    signal clk       : std_logic := '0';
    signal rst       : std_logic := '0';

    signal clear_acc : std_logic := '0';
    signal load_bias : std_logic := '0';

    signal bias_in   : signed(31 downto 0) := (others => '0');

    signal mac_en    : std_logic := '0';

    signal pixel_in  : unsigned(7 downto 0) := (others => '0');
    signal weight_in : signed(7 downto 0) := (others => '0');

    signal acc_out   : signed(31 downto 0);

    signal finished  : std_logic := '0';

begin

    -- Instantiate the MAC unit

    DUT : entity work.mac_unit
        port map (
            clk       => clk,
            rst       => rst,
            clear_acc => clear_acc,
            load_bias => load_bias,
            bias_in   => bias_in,
            mac_en    => mac_en,
            pixel_in  => pixel_in,
            weight_in => weight_in,
            acc_out   => acc_out
        );


    -- Clock generation

    clock_process : process
    begin

        while finished = '0' loop

            clk <= '0';
            wait for 5 ns;

            clk <= '1';
            wait for 5 ns;

        end loop;

        wait;

    end process;


    -- Test procedure

    stimulus_process : process
    begin

        -- Reset
        rst <= '1';

        wait for 22 ns;

        wait until falling_edge(clk);
        rst <= '0';


        -- Load initial bias = 5

        wait until falling_edge(clk);

        bias_in <= to_signed(5, 32);
        load_bias <= '1';

        wait until falling_edge(clk);

        load_bias <= '0';


        -- MAC operation 1: 5 * 2

        wait until falling_edge(clk);

        pixel_in <= to_unsigned(5, 8);
        weight_in <= to_signed(2, 8);

        mac_en <= '1';

        wait until rising_edge(clk);
        wait for 1 ns;

        assert acc_out = to_signed(15, 32)
            report "ERROR: MAC operation 1 failed"
            severity error;


        -- MAC operation 2: 3 * (-4)

        wait until falling_edge(clk);

        pixel_in <= to_unsigned(3, 8);
        weight_in <= to_signed(-4, 8);

        wait until rising_edge(clk);
        wait for 1 ns;

        assert acc_out = to_signed(3, 32)
            report "ERROR: MAC operation 2 failed"
            severity error;


        -- MAC operation 3: 7 * 1

        wait until falling_edge(clk);

        pixel_in <= to_unsigned(7, 8);
        weight_in <= to_signed(1, 8);

        wait until rising_edge(clk);
        wait for 1 ns;

        assert acc_out = to_signed(10, 32)
            report "ERROR: MAC operation 3 failed"
            severity error;


        -- Test accumulator clear

        wait until falling_edge(clk);

        mac_en <= '0';
        clear_acc <= '1';

        wait until falling_edge(clk);

        clear_acc <= '0';

        wait for 1 ns;

        assert acc_out = to_signed(0, 32)
            report "ERROR: Accumulator clear failed"
            severity error;


        report "ALL MAC TESTS PASSED!" severity note;

        finished <= '1';

        wait;

    end process;

end architecture simulation;
