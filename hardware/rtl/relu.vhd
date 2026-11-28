
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity relu is

    port (
        data_in  : in  signed(31 downto 0);
        data_out : out signed(31 downto 0)
    );

end entity relu;


architecture RTL of relu is

begin

    process(data_in)

    begin

        if data_in(31) = '1' then

            -- Negative input
            data_out <= (others => '0');

        else

            -- Positive or zero input
            data_out <= data_in;

        end if;

    end process;

end architecture RTL;
