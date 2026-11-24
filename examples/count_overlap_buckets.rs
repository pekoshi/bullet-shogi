use std::{fs::File, io::{Read, BufReader}};
use std::collections::HashSet;
fn main(){
 let root=r"E:\shogi\p2shogi\runs\profile15_overlap_audit";
 let mut overlap:u64=0; let mut unique:u64=0; let mut existing:u64=0; let mut candidate:u64=0;
 for i in 0..4096usize {
  let ep=format!(r"{}\existing\b{:04}.bin",root,i); let cp=format!(r"{}\candidate\b{:04}.bin",root,i);
  let mut set:HashSet<[u8;16]>=HashSet::new();
  if let Ok(f)=File::open(&ep){ let mut r=BufReader::new(f); let mut x=[0u8;16]; while r.read_exact(&mut x).is_ok(){set.insert(x); existing+=1;} }
  if let Ok(f)=File::open(&cp){ let mut r=BufReader::new(f); let mut x=[0u8;16]; while r.read_exact(&mut x).is_ok(){candidate+=1; if set.contains(&x){overlap+=1}else{unique+=1;} } }
  if i%256==0 { eprintln!("bucket={} existing={} candidate={} overlap={} unique={}",i,existing,candidate,overlap,unique); }
 }
 println!("existing_hash_records={} candidate_hash_records={} overlap_records={} candidate_hash_unique_vs_existing={}",existing,candidate,overlap,unique);
} 
