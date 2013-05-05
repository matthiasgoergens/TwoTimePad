{-# LANGUAGE OverloadedStrings #-}
import Data.HMM
import Control.Monad
-- import Data.Array
import Data.Array.Unboxed
import System.IO

import System.IO.MMap

import Data.Char
import Data.String
import qualified Data.ByteString.Char8 as BS

corpus :: IO (Array Int Char)
corpus = do b <- mmapFileByteString "corpus" Nothing
            let l = BS.length b
            return $ listArray (0,l-1) $ BS.unpack b

-- listArray

alpha = BS.unpack $ " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()"
--        " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()"


alphaX = [0..length alpha - 1]

myMM = simpleMM alpha 3

mm c = baumWelch myMM c 1

main = print . mm =<< corpus