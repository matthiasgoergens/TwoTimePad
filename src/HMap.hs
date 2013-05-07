{-# LANGUAGE FlexibleContexts #-}
-- module HMap where

import Prelude hiding (length, map, reverse)
import Control.Monad

import Data.Vector.Storable.MMap
-- import Data.Vector
import Foreign.Storable
import qualified Data.Vector.Storable as I
import Data.Vector
import System.Environment
import Data.Int
import Data.Char

import Common

g :: FilePath -> IO (Vector Int8)
g fp = fmap convert $ unsafeMMapVector fp Nothing

gc :: Vector Int8 -> Vector Char
gc = map (chr . fromIntegral)

f :: FilePath -> IO String
f fp = (fmap.fmap) (chr . fromIntegral) . fmap toList $ g fp

slicer :: Int -> Vector a -> Vector (Vector a)
slicer n v = generate (l - n + 1) (\i -> slice i n v)
  where l = length v

rev = reverse . map reverse

main = do
  (s:_) <- getArgs
--   return s
  x <- fmap (join . fmap toList . toList . rev . slicer 1 . gc) (g s)
  print x