
import Data.Char
import qualified Data.Set as S

import Common

fil :: String -> String
fil = uniqueSpace . filter isOK . map toLower . map toSpace
isOK :: Char -> Bool
isOK x = S.member x (S.fromList $ alpha)
toSpace :: Char -> Char
toSpace x | elem x " \t\n\r" = ' '
          | otherwise = x

uniqueSpace :: String -> String
uniqueSpace string = foldr op (const "") string True
  where op ' ' rest True = rest True
        op ' ' rest False = ' ' : rest True
        op char rest _ = char : rest False

main :: IO ()
main = putStr . fil =<< getContents
