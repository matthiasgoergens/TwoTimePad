import Data.String
import Data.List
import Data.Char
import qualified Data.Set as S

alpha = map toLower " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()"


fil = uniqueSpace . filter isOK . map toLower . map toSpace
isOK x = S.member x (S.fromList $ alpha)
toSpace x | elem x " \t\n\r" = ' '
          | otherwise = x

uniqueSpace string = foldr op (const "") string True
  where op ' ' rest True = rest True
        op ' ' rest False = ' ' : rest True
        op char rest _ = char : rest False

main = putStr . fil =<< getContents
