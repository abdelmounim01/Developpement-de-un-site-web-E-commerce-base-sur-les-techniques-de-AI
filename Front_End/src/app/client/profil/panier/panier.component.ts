import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../../../service/ImplementationService/projet.service'
import { Metauser } from 'src/app/service/Interfaces/metauser';

@Component({
  selector: 'app-panier',
  templateUrl: './panier.component.html',
  styleUrls: ['./panier.component.css']
})
export class PanierComponent implements OnInit {
  metauser:Metauser[]
  metauserlike:Metauser[]
  metauserrating:Metauser[]
  Metauserpanier:Metauser[]
  UserId:number = JSON.parse(localStorage.getItem('user')).userId
  meta:Metauser[];
  constructor(private projetService: ProjetService) { }

  
  ngOnInit(): void {
    this.getMetauser(this.UserId)
  }
  getMetauser (UserId): void {
    this.projetService.getMetauser(UserId).subscribe(metauser => (this.metauser = metauser))
  }
  getMetausers (): void {
    this.projetService.getMetausers().subscribe(meta => (this.meta = meta))
  }
  deletepanier (idproduct:number): void {

    this.projetService
    .getMetausers()
    .subscribe((data: Metauser[]) => {
      this.meta = data;
    const newMeta: Metauser = { userId:this.UserId , produitId : idproduct } as Metauser
    this.projetService.deletepanier(idproduct,this.UserId).subscribe(() => this.getMetausers())
    })
  }
  deleteSelf(meta:Metauser) {
    this.metauser = this.metauser.filter(h => h !== meta)
    /* this.metauser.push(meta) hadi ida bghina nzido*/ 
}

}
