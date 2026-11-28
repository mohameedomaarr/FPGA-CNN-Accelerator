
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity tb_relu is
end entity tb_relu;


architecture simulation of tb_relu is

    signal data_in  : signed(31 downto 0);
    signal data_out : signed(31 downto 0);

begin

    -- Instantiate the ReLU unit

    DUT : entity work.relu
        port map (
            data_in  => data_in,
            data_out => data_out
        );


    -- Test process

    stimulus_process : process

    begin

        -- Test 1: Positive input

        data_in <= to_signed(15, 32);

        wait for 10 ns;

        assert data_out = to_signed(15, 32)
            report "ERROR: Positive input test failed"
            severity error;


        -- Test 2: Negative input

        data_in <= to_signed(-8, 32);

        wait for 10 ns;

        assert data_out = to_signed(0, 32)
            report "ERROR: Negative input test failed"
            severity error;


        -- Test 3: Zero input

        data_in <= to_signed(0, 32);

        wait for 10 ns;

        assert data_out = to_signed(0, 32)
            report "ERROR: Zero input test failed"
            severity error;


        -- Test 4: Large positive input

        data_in <= to_signed(1000000, 32);

        wait for 10 ns;

        assert data_out = to_signed(1000000, 32)
            report "ERROR: Large positive input test failed"
            severity error;


        -- Test 5: Large negative input

        data_in <= to_signed(-1000000, 32);

        wait for 10 ns;

        assert data_out = to_signed(0, 32)
            report "ERROR: Large negative input test failed"
            severity error;


        report "ALL RELU TESTS PASSED!" severity note;

        wait;

    end process;

end architecture simulation;
