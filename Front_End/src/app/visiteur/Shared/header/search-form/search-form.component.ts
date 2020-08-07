import { Component, OnInit , ViewChild} from '@angular/core';
import { ProjetService } from '../../../../service/ImplementationService/projet.service'
import { Router } from '@angular/router'
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { HttpClient  } from '@angular/common/http'
import { AuthService } from 'src/app/service/auth/auth.service';
import { Task } from 'src/app/service/Interfaces/task';
import { PriceService } from 'src/app/service/PriceService/price.service';

@Component({
  selector: 'app-search-form-visiteur',
  templateUrl: './search-form.component.html',
  styleUrls: ['./search-form.component.css']
})
export class SearchFormvisiteurComponent implements OnInit {
  img:any
  reponse:any
  form: FormGroup;
  formtxt: FormGroup;
  class:any
  someVarName = null
  tasks:Task[]
  p: number=1
  message:string;
  isHidden:boolean = true;
  constructor(private price: PriceService ,private http: HttpClient,private projetService: ProjetService,private _router: Router,private authService: AuthService,private formBuilder: FormBuilder) { }

  ngOnInit() {
    this.price.currentMessage.subscribe(message => this.message = message) ; 
    this.form = this.formBuilder.group({
      file: [null, Validators.required]
    });
    this.formtxt = this.formBuilder.group({
      txt : [null, Validators.required],
    });
    console.log(this.tasks)
    this.someVarName = localStorage.getItem("img")

    console.log(this.someVarName)
    if(this.someVarName != null){
    this.projetService
  .images(this.someVarName)
  .subscribe((data: any) => {
    console.log(data.class)
    if(data.class == 'Literature'){
      this._router.navigate(['api/visiteur/Books/literature'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Families Relationship'){
      this._router.navigate(['api/visiteur/Books/families & relationship'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Fantasy'){
      this._router.navigate(['api/visiteur/Books/fantasy'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Children'){
      this._router.navigate(['api/visiteur/Books/children'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Classics'){
      this._router.navigate(['api/visiteur/Books/classics'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Crime'){
      this._router.navigate(['api/visiteur/Books/crime'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Short Stories'){
      this._router.navigate(['api/visiteur/Books/short stories'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Young Adults'){
      this._router.navigate(['api/visiteur/Books/young adults'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Romance'){
      this._router.navigate(['api/visiteur/Books/romance'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Science Fiction'){
      this._router.navigate(['api/visiteur/Books/science fiction'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'General'){
      this._router.navigate(['api/visiteur/Books/general'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Others'){
      this._router.navigate(['api/visiteur/Books/others'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Belts'){
      this._router.navigate(['api/visiteur/Accessories/belts'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Jewellery'){
      this._router.navigate(['api/visiteur/Accessories/jewellery'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Eyewear Accessories'){
      this._router.navigate(['api/visiteur/Accessories/eyewea'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Bags Accessories'){
      this._router.navigate(['api/visiteur/Accessories/bags'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Headwear Accessories'){
      this._router.navigate(['api/visiteur/Accessories/headwear'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Wallets'){
      this._router.navigate(['api/visiteur/Accessories/wallets'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Cufflinks'){
      this._router.navigate(['api/visiteur/Accessories/cufflinks'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Gloves'){
      this._router.navigate(['api/visiteur/Accessories/gloves'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Watches Accessories'){
      this._router.navigate(['api/visiteur/Accessories/watches'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Gift'){
      this._router.navigate(['api/visiteur/Accessories/accessory gift set'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Scarves'){
      this._router.navigate(['api/visiteur/Apparel/scarves'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Innerwear'){
      this._router.navigate(['api/visiteur/Apparel/innerwear'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Bottomwear'){
      this._router.navigate(['api/visiteur/Apparel/bottomwear'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Topwear'){
      this._router.navigate(['api/visiteur/Apparel/topwear'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Apparel Set'){
      this._router.navigate(['api/visiteur/Apparel/apparel set'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Dress'){
      this._router.navigate(['api/visiteur/Apparel/dress'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Ties'){
      this._router.navigate(['api/visiteur/Apparel/ties'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Stoles'){
      this._router.navigate(['api/visiteur/Apparel/stoles'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Loungewear Nightwear'){
      this._router.navigate(['api/visiteur/Apparel/loungewear and nightwear'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Saree'){
      this._router.navigate(['api/visiteur/Apparel/saree'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Socks'){
      this._router.navigate(['api/visiteur/Footwear/socks'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Shoes'){
      this._router.navigate(['api/visiteur/Footwear/shoes'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Flip Flops'){
      this._router.navigate(['api/visiteur/Footwear/flip flops'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Sandal'){
      this._router.navigate(['api/visiteur/Footwear/sandal'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Shoe Accessories'){
      this._router.navigate(['api/visiteur/Footwear/shoe accessories'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Fragrance'){
      this._router.navigate(['api/visiteur/Personnal Care/fragrance'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Lips'){
      this._router.navigate(['api/visiteur/Personnal Care/lips'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Eyes'){
      this._router.navigate(['api/visiteur/Personnal Care/eyes'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Hair'){
      this._router.navigate(['api/visiteur/Personnal Care/hair'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Makeup'){
      this._router.navigate(['api/visiteur/Personnal Care/makeup'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Bath Body'){
      this._router.navigate(['api/visiteur/Personnal Care/bath and body'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Skin Care'){
      this._router.navigate(['api/visiteur/Personnal Care/skin care'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Nails'){
      this._router.navigate(['api/visiteur/Personnal Care/nails'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Bags Sporting'){
      this._router.navigate(['api/visiteur/Sporting Goods/bags'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Headwear Sporting'){
      this._router.navigate(['api/visiteur/Sporting Goods/headwear'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Water Bottle'){
      this._router.navigate(['api/visiteur/Sporting Goods/water bottle'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Eyewear Sporting'){
      this._router.navigate(['api/visiteur/Sporting Goods/eyewear'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Watches Sportings'){
      this._router.navigate(['api/visiteur/Sporting Goods/watches'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Sport Equipement'){
      this._router.navigate(['api/visiteur/Sporting Goods/sports equipment'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Android'){
      this._router.navigate(['api/visiteur/Phones/android'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Honor'){
      this._router.navigate(['api/visiteur/Phones/honor'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Vivo'){
      this._router.navigate(['api/visiteur/Phones/vivo'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Sony'){
      this._router.navigate(['api/visiteur/Phones/sony'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Oppo'){
      this._router.navigate(['api/visiteur/Phones/oppo'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Redmi'){
      this._router.navigate(['api/visiteur/Phones/redmi'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Nokia'){
      this._router.navigate(['api/visiteur/Phones/nokia'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Oneplus'){
      this._router.navigate(['api/visiteur/Phones/oneplus'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Apple'){
      this._router.navigate(['api/visiteur/Phones/apple'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Samsung'){
      this._router.navigate(['api/visiteur/Phones/samsung'])
      localStorage.removeItem('img')
    }
    else if(data.class == 'Huawei'){
      this._router.navigate(['api/visiteur/Phones/huawei'])
      localStorage.removeItem('img')
    }
  
  });
  }
  }
  @ViewChild('fileInput') fileInput;
  uploadFile() {
    const files: FileList = this.fileInput.nativeElement.files;
    if (files.length === 0) {
      return;
    };
    this.projetService.parseTable(files).subscribe((data: any) => {
      console.log(data);
    });
}

/*sent(fileInput:string){
  this.img = fileInput.replace("C:\\fakepath\\", "")
  console.log(this.someVarName)
  
  
}*/

image(fileInput:string): void {
  this.uploadFile()
  
  this.img = fileInput.replace("C:\\fakepath\\", "")
  var someVarName = this.img;
  localStorage.setItem("img", someVarName);
  console.log(this.img)
  this.projetService.images(this.img).subscribe(reponse => (this.reponse = reponse))
  /*this.sent(fileInput)*/
  window.location.reload();
}

searchproduct(txt:string): void {
  
  console.log(txt)
  this.projetService.search(txt).subscribe(tasks => (this.tasks = tasks))
  /*this.sent(fileInput)*/
  this.isHidden = false;
}
}
