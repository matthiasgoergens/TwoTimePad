
import Data.Char
import qualified Data.Set as S

import Common


-- TODO: Consider bytestring..
fil :: String -> String
fil = uniqueSpace . filter isOK . map toLower . map replace
isOK :: Char -> Bool
isOK x = S.member x (S.fromList $ alpha)
replace :: Char -> Char
replace '"' = '\''
replace '!' = '.'
replace x | elem x "\t\n\r" = ' '
replace x = x

uniqueSpace :: String -> String
uniqueSpace string = foldr op (const "") string True
  where op ' ' rest True = rest True
        op ' ' rest False = ' ' : rest True
        op char rest _ = char : rest False

main :: IO ()
main = putStr . fil =<< getContents
