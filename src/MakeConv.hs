import Common
import Control.Monad

f a n = putStr $ "frChr '" ++ [a] ++ "' = " ++ show n ++ "\n"

t a n = putStr $ "toChr " ++ show n ++ " = '" ++ [a] ++ "'\n"

main = do
  zipWithM f alpha [0..]
  zipWithM t alpha [0..]    
